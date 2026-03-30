import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Restomart Smart Search",
    page_icon="🍽️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.main { background: #0f0f0f; }
[data-testid="stAppViewContainer"] { background: #0f0f0f; color: #f0ece4; }
[data-testid="stSidebar"] { background: #1a1a1a; border-right: 1px solid #2a2a2a; }

.product-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.product-card:hover { border-color: #e8c547; }

.match-badge {
    background: #e8c547;
    color: #0f0f0f;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.4rem 0.8rem;
    border-radius: 8px;
    display: inline-block;
}
.category-tag {
    background: #2a2a2a;
    color: #888;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 0.5rem;
}
.price-tag {
    color: #e8c547;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
}
.rank-num {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #2a2a2a;
    line-height: 1;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL + DATABASE (cached = loads once)
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    # This model runs 100% locally on your laptop — no API key needed
    # It converts text → 384 numbers (coordinates in meaning-space)
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def setup_database():
    model = load_model()

    # ── Restomart sample catalog ──────────────────────────────────
    # In production: replace with your real Restomart SKUs
    products = [
        {"id": "p001", "name": "Anchor UHT Full Cream Milk",
         "description": "Full cream UHT milk ideal for coffee, tea, baking, and creamy pasta sauces. Long shelf life. 1L cartons.",
         "category": "Dairy", "price": "₹85/L"},
        {"id": "p002", "name": "President Unsalted Butter",
         "description": "Premium French unsalted butter for baking, sautéing, and finishing sauces. Rich flavor, professional grade.",
         "category": "Dairy", "price": "₹320/250g"},
        {"id": "p003", "name": "Amul Fresh Cream",
         "description": "Fresh cooking cream for gravies, pasta, soups, and desserts. 25% fat content. Best for Indian and continental cuisine.",
         "category": "Dairy", "price": "₹55/200ml"},
        {"id": "p004", "name": "Kagome Tomato Puree",
         "description": "Concentrated tomato puree for pizza sauce, pasta, curries. No preservatives. Restaurant pack 2.8kg.",
         "category": "Sauces", "price": "₹420/tin"},
        {"id": "p005", "name": "Borges Extra Virgin Olive Oil",
         "description": "Cold-pressed Spanish olive oil for dressings, marinades, and Mediterranean cooking. Fruity aroma.",
         "category": "Oils & Fats", "price": "₹680/L"},
        {"id": "p006", "name": "McCain Frozen French Fries",
         "description": "Straight-cut frozen fries for fast food operations. Consistent size, crispy texture. 2.5kg pack.",
         "category": "Frozen", "price": "₹380/pack"},
        {"id": "p007", "name": "Sysco Classic Chicken Stock",
         "description": "Rich chicken broth base for soups, risottos, braises, and sauces. Low sodium, professional grade. 1L tetra.",
         "category": "Stocks", "price": "₹210/L"},
        {"id": "p008", "name": "Parmesan Grana Padano Block",
         "description": "Aged Italian hard cheese for pasta, risotto, and salads. Nutty, umami-rich flavor. 1kg block.",
         "category": "Cheese", "price": "₹1,850/kg"},
        {"id": "p009", "name": "De Cecco Penne Rigate",
         "description": "Bronze-die cut Italian penne pasta. Holds sauces well. Ideal for baked pasta, arrabbiata, cream sauces.",
         "category": "Dry Goods", "price": "₹280/500g"},
        {"id": "p010", "name": "Keya Mixed Herbs",
         "description": "Blend of oregano, basil, thyme, rosemary. Essential for Italian, Mediterranean, and continental cooking.",
         "category": "Spices", "price": "₹95/jar"},
    ]

    # Build ChromaDB collection (stored in memory locally)
    client = chromadb.Client()
    collection = client.create_collection(
        name="restomart_products",
        metadata={"hnsw:space": "cosine"}
    )

    # Convert every product description → embedding → store in ChromaDB
    # This is the "indexing" step — done once at startup
    descriptions = [p["description"] for p in products]
    embeddings = model.encode(descriptions).tolist()

    collection.add(
        ids=[p["id"] for p in products],
        embeddings=embeddings,
        documents=descriptions,
        metadatas=[{"name": p["name"], "category": p["category"], "price": p["price"]} for p in products]
    )

    return collection, model


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

col_title, _ = st.columns([2, 1])
with col_title:
    st.markdown("# 🍽️ Restomart Smart Search")
    st.markdown("<p style='color:#888; margin-top:-0.5rem;'>Search by meaning — not just keywords</p>", unsafe_allow_html=True)

st.markdown("---")

# Load everything
with st.spinner("Loading AI model (first run only — ~30 seconds)..."):
    collection, model = setup_database()

# ── Search bar ───────────────────────────────
left, right = st.columns([3, 1])
with left:
    query = st.text_input(
        "",
        placeholder="Describe what you need — e.g. 'something creamy for white pasta sauce'",
        label_visibility="collapsed"
    )
with right:
    num_results = st.selectbox("Results", [3, 5, 10], label_visibility="collapsed")

# ── Suggested searches ───────────────────────
st.markdown("<p style='color:#555; font-size:0.8rem;'>Try: &nbsp;&nbsp;"
            "<b style='color:#888'>creamy white sauce base</b> &nbsp;·&nbsp; "
            "<b style='color:#888'>umami flavour booster</b> &nbsp;·&nbsp; "
            "<b style='color:#888'>fat for sautéing</b> &nbsp;·&nbsp; "
            "<b style='color:#888'>carbs for Italian dish</b> &nbsp;·&nbsp; "
            "<b style='color:#888'>base for hearty soup</b>"
            "</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SEARCH LOGIC
# ─────────────────────────────────────────────

if query:
    with st.spinner("Finding best matches..."):
        # Step 1: Convert buyer's query → embedding
        query_embedding = model.encode([query]).tolist()

        # Step 2: ChromaDB finds closest product embeddings
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=num_results
        )

    st.markdown(f"### Results for *\"{query}\"*")
    st.markdown("<br>", unsafe_allow_html=True)

    for i, (doc, meta, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        similarity = round((1 - distance) * 100, 1)

        st.markdown(f"""
        <div class="product-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div style="flex:1;">
                    <span class="rank-num">0{i+1}</span>
                    <span class="category-tag" style="margin-left:0.5rem;">{meta['category']}</span><br>
                    <span style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700;">{meta['name']}</span><br>
                    <span style="color:#aaa; font-size:0.9rem;">{doc}</span><br><br>
                    <span class="price-tag">{meta['price']}</span>
                </div>
                <div style="text-align:right; padding-left:1rem;">
                    <span class="match-badge">{similarity}%</span><br>
                    <span style="color:#555; font-size:0.75rem;">match</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 How This Works")
    st.markdown("""
Every product description is converted into **384 numbers** — coordinates in meaning-space.

When you search, your query becomes coordinates too. We find the closest products.

**No keywords. Pure meaning.**

---
**Stack**
- `sentence-transformers` — local embedding model
- `ChromaDB` — vector database
- `Streamlit` — UI

**Cost: ₹0**

---
Built by Kathiravan · Day 2 of AI Sprint
    """)