import re
import difflib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import networkx as nx
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components


# =============================
# KG 初始化
# =============================
def build_kg() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    # 节点：实体 + 类型
    nodes = [
        ("Alice", {"type": "Person", "aliases": ["alice", "艾丽丝", "阿丽丝"]}),
        ("Bob", {"type": "Person", "aliases": ["bob", "鲍勃"]}),
        ("OpenAI", {"type": "Org", "aliases": ["openai", "open ai", "Open AI", "开放AI"]}),
        ("GPT-5", {"type": "Model", "aliases": ["gpt5", "gpt-5", "GPT5"]}),
        ("SanFrancisco", {"type": "City", "aliases": ["san francisco", "sf", "旧金山"]}),
        ("Tokyo", {"type": "City", "aliases": ["tokyo", "东京"]}),
    ]
    for n, attrs in nodes:
        G.add_node(n, **attrs)

    def add_rel(h, r, t, **attrs):
        G.add_edge(h, t, relation=r, **attrs)

    add_rel("Alice", "works_at", "OpenAI", since=2024)
    add_rel("Bob", "works_at", "OpenAI", since=2025)
    add_rel("OpenAI", "develops", "GPT-5")
    add_rel("OpenAI", "located_in", "SanFrancisco")
    add_rel("Bob", "lives_in", "Tokyo")
    add_rel("Alice", "knows", "Bob", weight=0.8)

    return G


# =============================
# 实用：图查询
# =============================
def tails(G: nx.MultiDiGraph, head: str, relation: str) -> List[Tuple[str, Dict[str, Any]]]:
    if head not in G:
        return []
    out = []
    for _, t, data in G.out_edges(head, data=True):
        if data.get("relation") == relation:
            out.append((t, dict(data)))
    return out


def heads(G: nx.MultiDiGraph, tail: str, relation: str) -> List[Tuple[str, Dict[str, Any]]]:
    if tail not in G:
        return []
    out = []
    for h, _, data in G.in_edges(tail, data=True):
        if data.get("relation") == relation:
            out.append((h, dict(data)))
    return out


def neighbors_1hop(G: nx.MultiDiGraph, ent: str) -> List[Tuple[str, str, str]]:
    """返回 (head, relation, tail) 一跳邻接（出边）"""
    res = []
    if ent not in G:
        return res
    for _, t, data in G.out_edges(ent, data=True):
        res.append((ent, data.get("relation", "related_to"), t))
    return res


def two_hop(G: nx.MultiDiGraph, start: str, r1: str, r2: str) -> List[Tuple[str, str]]:
    ans = []
    for mid, _ in tails(G, start, r1):
        for end, _ in tails(G, mid, r2):
            ans.append((mid, end))
    return ans


# =============================
# 实体对齐：别名 + 模糊匹配
# =============================
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def build_alias_index(G: nx.MultiDiGraph) -> Dict[str, str]:
    """
    alias_index[alias_norm] = canonical_entity
    """
    idx = {}
    for n, attrs in G.nodes(data=True):
        idx[normalize_text(n)] = n
        for a in attrs.get("aliases", []) or []:
            idx[normalize_text(str(a))] = n
    return idx


def resolve_entity(G: nx.MultiDiGraph, mention: str, cutoff: float = 0.72) -> Tuple[Optional[str], str]:
    """
    返回 (canonical, debug_reason)
    - 先 exact alias
    - 再 difflib 模糊匹配
    """
    m = normalize_text(mention)
    alias_idx = build_alias_index(G)

    if m in alias_idx:
        return alias_idx[m], f"alias exact: {mention} -> {alias_idx[m]}"

    # fuzzy: 在所有 alias key 上做匹配
    keys = list(alias_idx.keys())
    best = difflib.get_close_matches(m, keys, n=1, cutoff=cutoff)
    if best:
        return alias_idx[best[0]], f"alias fuzzy: {mention} -> {alias_idx[best[0]]} (via '{best[0]}')"

    return None, f"unresolved: {mention}"


# =============================
# QA：意图 + 槽位解析（规则版 + 更鲁棒）
# =============================
REL_SYNONYMS = {
    "works_at": ["works_at", "work at", "works at", "在", "就职", "工作于", "工作在", "供职", "公司"],
    "located_in": ["located_in", "located in", "在", "位于", "坐落", "地点", "哪里"],
    "develops": ["develops", "develop", "研发", "开发", "做了", "推出"],
    "knows": ["knows", "know", "认识", "熟悉", "朋友"],
    "lives_in": ["lives_in", "lives in", "住在", "居住", "生活在"],
}


def resolve_relation(text: str) -> Optional[str]:
    t = normalize_text(text)
    # 优先如果直接写了标准关系名
    for r in REL_SYNONYMS.keys():
        if re.search(rf"\b{re.escape(r)}\b", t):
            return r
    # 否则按同义词扫描
    for r, syns in REL_SYNONYMS.items():
        for s in syns:
            if s and normalize_text(s) in t:
                return r
    return None


@dataclass
class ParsedQuery:
    intent: str
    slots: Dict[str, str]


def parse_question(q: str) -> ParsedQuery:
    raw = q.strip()

    # 多条件：在X的人里谁在Y工作？
    m = re.fullmatch(r"在(.+)的人里谁在(.+)工作\??", raw)
    if m:
        return ParsedQuery("who_works_at_in_city", {"city": m.group(1).strip(), "org": m.group(2).strip()})

    # 谁在X工作
    m = re.fullmatch(r"谁在(.+)工作\??", raw)
    if m:
        return ParsedQuery("who_works_at", {"org": m.group(1).strip()})

    # X在哪里（组织 located_in / 人 lives_in）
    m = re.fullmatch(r"(.+)在哪里\??", raw)
    if m:
        return ParsedQuery("where_is", {"entity": m.group(1).strip()})

    # X开发了什么
    m = re.fullmatch(r"(.+)开发了什么\??", raw)
    if m:
        return ParsedQuery("what_develops", {"entity": m.group(1).strip()})

    # X认识谁
    m = re.fullmatch(r"(.+)认识谁\??", raw)
    if m:
        return ParsedQuery("who_does_know", {"entity": m.group(1).strip()})

    # 两跳：X工作的公司在哪里
    m = re.fullmatch(r"(.+)工作的公司在哪里\??", raw)
    if m:
        return ParsedQuery("where_company_of_person", {"person": m.group(1).strip()})

    # 一跳邻居：X的一跳邻居 / X有哪些关系
    m = re.fullmatch(r"(.+)的一跳邻居\??", raw)
    if m:
        return ParsedQuery("one_hop", {"entity": m.group(1).strip()})

    # 三元组查询：A 关系 B（关系可以是标准英文或中文同义词）
    m = re.fullmatch(r"(.+?)\s+(.+?)\s+(.+)", raw)
    if m:
        return ParsedQuery("triple_query", {"h": m.group(1).strip(), "r": m.group(2).strip(), "t": m.group(3).strip()})

    return ParsedQuery("unknown", {"raw": raw})


def answer(G: nx.MultiDiGraph, q: str) -> Tuple[str, List[str]]:
    """
    返回 (answer_text, debug_lines)
    """
    pq = parse_question(q)
    dbg = [f"intent={pq.intent}", f"slots={pq.slots}"]

    def R(name: str) -> Optional[str]:
        ent, why = resolve_entity(G, name)
        dbg.append(why)
        return ent

    if pq.intent == "who_works_at_in_city":
        city = R(pq.slots["city"])
        org = R(pq.slots["org"])
        if not city or not org:
            return "我没能把问题里的实体对齐到图谱节点（可能是拼写/别名没收录）。", dbg

        # 候选：works_at -> org 的所有人
        people = [h for h, _ in heads(G, org, "works_at")]
        # 过滤：lives_in == city
        hits = []
        for p in people:
            if any(t == city for t, _ in tails(G, p, "lives_in")):
                hits.append(p)
        if not hits:
            return f"图里没找到“住在 {city} 且在 {org} 工作”的人。", dbg
        return f"住在 {city} 且在 {org} 工作的人：{', '.join(sorted(set(hits)))}", dbg

    if pq.intent == "who_works_at":
        org = R(pq.slots["org"])
        if not org:
            return "我没能识别你说的组织/公司是谁。", dbg
        hs = heads(G, org, "works_at")
        if not hs:
            return f"我没在图里找到谁在 {org} 工作。", dbg
        return f"在 {org} 工作的人：{', '.join(sorted(set([h for h, _ in hs])))}", dbg

    if pq.intent == "where_is":
        ent = R(pq.slots["entity"])
        if not ent:
            return "我没能识别你问的实体是谁。", dbg

        locs = tails(G, ent, "located_in")
        if locs:
            return f"{ent} 位于：{', '.join(sorted(set([t for t, _ in locs])))}", dbg

        lives = tails(G, ent, "lives_in")
        if lives:
            return f"{ent} 居住在：{', '.join(sorted(set([t for t, _ in lives])))}", dbg

        return f"我没在图里找到 {ent} 的位置信息（located_in / lives_in）。", dbg

    if pq.intent == "what_develops":
        ent = R(pq.slots["entity"])
        if not ent:
            return "我没能识别你问的实体是谁。", dbg
        xs = tails(G, ent, "develops")
        if not xs:
            return f"我没在图里找到 {ent} 开发/研发了什么。", dbg
        return f"{ent} 开发：{', '.join(sorted(set([t for t, _ in xs])))}", dbg

    if pq.intent == "who_does_know":
        ent = R(pq.slots["entity"])
        if not ent:
            return "我没能识别你问的人是谁。", dbg
        xs = tails(G, ent, "knows")
        if not xs:
            return f"我没在图里找到 {ent} 认识谁。", dbg
        return f"{ent} 认识：{', '.join(sorted(set([t for t, _ in xs])))}", dbg

    if pq.intent == "where_company_of_person":
        person = R(pq.slots["person"])
        if not person:
            return "我没能识别你说的人是谁。", dbg
        hops = two_hop(G, person, "works_at", "located_in")
        if not hops:
            return f"我没在图里找到 {person} -> works_at -> located_in 的两跳路径。", dbg
        uniq = sorted(set(hops))
        return "；".join([f"{person} 在 {c} 工作，{c} 位于 {city}" for c, city in uniq]), dbg

    if pq.intent == "one_hop":
        ent = R(pq.slots["entity"])
        if not ent:
            return "我没能识别你问的实体是谁。", dbg
        hops = neighbors_1hop(G, ent)
        if not hops:
            return f"{ent} 没有出边关系（或不在图里）。", dbg
        lines = [f"({h}, {r}, {t})" for h, r, t in hops]
        return f"{ent} 的一跳出边：\n" + "\n".join(lines), dbg

    if pq.intent == "triple_query":
        h = R(pq.slots["h"])
        t = R(pq.slots["t"])
        r = resolve_relation(pq.slots["r"]) or normalize_text(pq.slots["r"])
        dbg.append(f"relation_resolved={r}")

        if not h or not t:
            return "我没能把三元组里的头/尾实体对齐到图谱节点。", dbg

        exists = False
        for _, tt, data in G.out_edges(h, data=True):
            if tt == t and data.get("relation") == r:
                exists = True
                break
        return ("是的，图里有这条关系。" if exists else "没有，我没在图里找到这条关系。"), dbg

    return (
        "我暂时没理解这个问题（目前还是规则解析）。\n"
        "你可以试试：\n"
        "- 谁在OpenAI工作\n"
        "- OpenAI在哪里 / Bob在哪里\n"
        "- OpenAI开发了什么\n"
        "- Alice认识谁\n"
        "- Alice工作的公司在哪里\n"
        "- 在Tokyo的人里谁在OpenAI工作\n"
        "- Alice的一跳邻居\n"
        "- Alice works_at OpenAI",
        dbg
    )


# =============================
# 可视化：PyVis
# =============================
def render_pyvis(G: nx.MultiDiGraph, height="680px") -> str:
    net = Network(height=height, width="100%", directed=True, notebook=False)

    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.2,
        spring_length=140,
        spring_strength=0.05,
        damping=0.4
    )

    for n, attrs in G.nodes(data=True):
        ntype = attrs.get("type", "Entity")
        aliases = attrs.get("aliases", [])
        title = f"{n}\n(type={ntype})"
        if aliases:
            title += "\naliases=" + ", ".join(map(str, aliases))
        net.add_node(n, label=n, title=title)

    for h, t, attrs in G.edges(data=True):
        r = attrs.get("relation", "related_to")
        extra = {k: v for k, v in attrs.items() if k != "relation"}
        title = f"{h} -[{r}]-> {t}"
        if extra:
            title += "\n" + "\n".join([f"{k}={v}" for k, v in extra.items()])
        net.add_edge(h, t, label=r, title=title, arrows="to")

    return net.generate_html()


# =============================
# 编辑：新增/删除
# =============================
def add_or_update_node(G: nx.MultiDiGraph, name: str, ntype: str, aliases_csv: str):
    name = name.strip()
    if not name:
        return
    aliases = [a.strip() for a in (aliases_csv or "").split(",") if a.strip()]
    if name not in G:
        G.add_node(name, type=ntype, aliases=aliases)
    else:
        # update
        G.nodes[name]["type"] = ntype or G.nodes[name].get("type", "Entity")
        # merge aliases
        old = set(G.nodes[name].get("aliases", []) or [])
        for a in aliases:
            old.add(a)
        G.nodes[name]["aliases"] = sorted(old)


def remove_node(G: nx.MultiDiGraph, name: str) -> bool:
    if name in G:
        G.remove_node(name)
        return True
    return False


def add_edge(G: nx.MultiDiGraph, h: str, r: str, t: str, props: Dict[str, Any]):
    if h not in G or t not in G:
        raise ValueError("head/tail 必须是已存在的节点。")
    G.add_edge(h, t, relation=r, **props)


def remove_edge_by_index(G: nx.MultiDiGraph, idx: int) -> bool:
    """
    按当前 edges 列表顺序删除第 idx 条边（给 UI 用）
    """
    edges = list(G.edges(keys=True, data=True))
    if idx < 0 or idx >= len(edges):
        return False
    h, t, k, _ = edges[idx]
    G.remove_edge(h, t, key=k)
    return True


# =============================
# Streamlit App
# =============================
st.set_page_config(page_title="KG Topology + QA (Editable)", layout="wide")
st.title("知识图谱：网络拓扑图 + 可编辑 + 基础问答")

# Session state: 图谱持久化
if "G" not in st.session_state:
    st.session_state.G = build_kg()

G: nx.MultiDiGraph = st.session_state.G

left, right = st.columns([1.15, 1])

with left:
    st.subheader("网络拓扑图（拖拽/缩放/悬停）")
    html = render_pyvis(G)
    components.html(html, height=720, scrolling=True)

    st.caption("提示：节点悬停可以看到 type/aliases；边悬停可以看到属性。")

with right:
    tabs = st.tabs(["查询问答", "编辑图谱", "调试/导出"])

    # -------- QA --------
    with tabs[0]:
        st.subheader("查询问答（KGQA-lite）")
        q = st.text_input("输入问题", value="在Tokyo的人里谁在OpenAI工作？")
        col1, col2 = st.columns([1, 1])
        with col1:
            run = st.button("查询", type="primary")
        with col2:
            show_dbg = st.toggle("显示解析调试", value=False)

        if run:
            ans, dbg = answer(G, q)
            st.markdown(ans)
            if show_dbg:
                st.code("\n".join(dbg))

        st.divider()
        st.markdown("**示例问法：**")
        st.markdown(
            "- 谁在OpenAI工作？\n"
            "- OpenAI在哪里？ / Bob在哪里？\n"
            "- OpenAI开发了什么？\n"
            "- Alice认识谁？\n"
            "- Alice工作的公司在哪里？\n"
            "- 在Tokyo的人里谁在OpenAI工作？\n"
            "- Alice的一跳邻居？\n"
            "- Alice works_at OpenAI"
        )

    # -------- Edit --------
    with tabs[1]:
        st.subheader("编辑图谱")

        st.markdown("### 1) 新增/更新节点")
        with st.form("node_form", clear_on_submit=False):
            n = st.text_input("节点名（canonical）", value="Anthropic")
            ntype = st.selectbox("节点类型", ["Person", "Org", "City", "Model", "Entity"], index=1)
            aliases = st.text_input("aliases（逗号分隔，可空）", value="anthropic, 安全AI")
            submitted = st.form_submit_button("保存节点")
            if submitted:
                add_or_update_node(G, n, ntype, aliases)
                st.success(f"已保存节点：{n}")

        st.markdown("### 2) 新增关系（边）")
        node_list = sorted(list(G.nodes()))
        with st.form("edge_form", clear_on_submit=False):
            h = st.selectbox("Head", node_list, index=0)
            r = st.selectbox("Relation", ["works_at", "located_in", "develops", "knows", "lives_in"], index=0)
            t = st.selectbox("Tail", node_list, index=min(1, len(node_list)-1))
            props_text = st.text_input("边属性（key=value, 逗号分隔，可空）", value="since=2026")
            submitted2 = st.form_submit_button("添加关系")
            if submitted2:
                props = {}
                if props_text.strip():
                    for kv in props_text.split(","):
                        kv = kv.strip()
                        if not kv:
                            continue
                        if "=" not in kv:
                            continue
                        k, v = kv.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        # 尝试转 int/float
                        if re.fullmatch(r"-?\d+", v):
                            v = int(v)
                        elif re.fullmatch(r"-?\d+\.\d+", v):
                            v = float(v)
                        props[k] = v
                try:
                    add_edge(G, h, r, t, props)
                    st.success(f"已添加：({h}, {r}, {t})")
                except Exception as e:
                    st.error(str(e))

        st.markdown("### 3) 删除节点/关系")
        c1, c2 = st.columns(2)

        with c1:
            del_node = st.selectbox("选择要删除的节点", ["(不删除)"] + node_list, index=0)
            if st.button("删除节点", type="secondary"):
                if del_node != "(不删除)" and remove_node(G, del_node):
                    st.success(f"已删除节点：{del_node}")
                else:
                    st.info("未删除（可能未选择或节点不存在）。")

        with c2:
            edges = list(G.edges(keys=True, data=True))
            edge_labels = []
            for i, (hh, tt, kk, dd) in enumerate(edges):
                edge_labels.append(f"[{i}] ({hh}, {dd.get('relation')}, {tt}) props={{{', '.join([f'{k}={v}' for k, v in dd.items() if k!='relation'])}}}")
            del_edge = st.selectbox("选择要删除的边（按索引）", ["(不删除)"] + edge_labels, index=0)
            if st.button("删除边", type="secondary"):
                if del_edge != "(不删除)":
                    idx = int(re.search(r"\[(\d+)\]", del_edge).group(1))
                    ok = remove_edge_by_index(G, idx)
                    st.success("已删除边。" if ok else "删除失败（索引无效）。")
                else:
                    st.info("未删除（未选择）。")

        st.caption("删除/新增会立刻影响左侧拓扑图与问答结果。")

    # -------- Debug/Export --------
    with tabs[2]:
        st.subheader("调试/导出")
        st.markdown("### 当前三元组（head, relation, tail）")
        triples = []
        for h, t, data in G.edges(data=True):
            triples.append((h, data.get("relation"), t))
        st.code("\n".join([f"({h}, {r}, {t})" for h, r, t in triples]))

        st.markdown("### 导出：GraphML（方便给 Gephi / 其它工具）")
        if st.button("生成 GraphML 文本（复制保存为 .graphml）"):
            # networkx 写 graphml 需要文件对象，Streamlit 用临时字符串写法
            import io
            buf = io.BytesIO()
            nx.write_graphml(G, buf)
            txt = buf.getvalue().decode("utf-8", errors="ignore")
            st.text_area("GraphML", value=txt, height=260)

        st.markdown("### 快捷：重置回 demo 图谱")
        if st.button("重置图谱", type="secondary"):
            st.session_state.G = build_kg()
            st.success("已重置。刷新页面或继续操作即可。")