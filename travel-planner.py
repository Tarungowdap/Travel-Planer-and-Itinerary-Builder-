# app_bart.py
# Travel Planner & Itinerary Builder with BART (RAG + Agent)
# Run once: pip install flask flask-cors transformers torch requests
# Start: python app_bart.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from uuid import uuid4
from datetime import datetime
import requests, re, time, random
from transformers import pipeline

# ---- Models (BART) ----
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")
mode_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", tokenizer="facebook/bart-large-mnli")
CANDIDATE_MODES = ["budget-friendly", "adventure", "luxury"]

# ---- App ----
app = Flask(__name__)
CORS(app)
USERS = {}

# ---- Utils ----
def new_id(prefix="id"):
    return f"{prefix}_{uuid4().hex[:8]}"

def sanitize_interests(interests):
    if not interests:
        return []
    if isinstance(interests, str):
        return [s.strip().lower() for s in re.split(r"[,\|/;]+", interests) if s.strip()]
    return [str(x).strip().lower() for x in interests if str(x).strip()]

def ensure_user(user_id):
    if user_id not in USERS:
        USERS[user_id] = {"itinerary": None, "activities": {}}

def err(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code

# ---- Simple cache ----
CACHE = {}
def get_cache(key, ttl=3600):
    v = CACHE.get(key)
    if not v: return None
    ts, data = v
    if time.time() - ts > ttl:
        CACHE.pop(key, None)
        return None
    return data

def set_cache(key, data):
    CACHE[key] = (time.time(), data)

# ---- Wikipedia RAG ----
WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

def wiki_search_titles(query, limit=8):
    key = f"search:{query}:{limit}"
    c = get_cache(key)
    if c: return c
    try:
        r = requests.get(WIKI_SEARCH, params={
            "action": "query", "list": "search", "srsearch": query, "srlimit": limit, "utf8": 1, "format": "json"
        }, timeout=6).json()
        titles = [it["title"] for it in r.get("query", {}).get("search", [])]
    except Exception:
        titles = []
    set_cache(key, titles)
    return titles

def wiki_fetch_text_for_title(title):
    key = f"summary:{title}"
    c = get_cache(key)
    if c: return c
    try:
        r = requests.get(WIKI_SUMMARY.format(requests.utils.quote(title)), timeout=6)
        if r.status_code != 200:
            set_cache(key, (None, None))
            return None, None
        j = r.json()
        extract = j.get("extract", "") or ""
        desc = j.get("description", "") or ""
        text = (desc + ". " if desc else "") + extract
        url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(j.get('title',''))}"
        set_cache(key, (text.strip(), url))
        return text.strip(), url
    except Exception:
        set_cache(key, (None, None))
        return None, None

def chunk_text(text, max_chars=2500):
    if not text: return []
    text = text.strip()
    if len(text) <= max_chars: return [text]
    chunks = []; start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        cut = text.rfind(".", start, end)
        if cut == -1 or cut <= start + 200: cut = end
        chunks.append(text[start:cut].strip())
        start = cut + 1
    return chunks

def bart_summarize(text, max_len=110, min_len=30):
    if not text: return ""
    chunks = chunk_text(text)
    parts = []
    for ch in chunks:
        try:
            out = summarizer(ch, max_length=max_len, min_length=min_len, do_sample=False)
            parts.append(out[0]["summary_text"])
        except Exception:
            parts.append(ch[:max_len])
    if len(parts) > 1:
        try:
            final = summarizer(" ".join(parts), max_length=max_len, min_length=min_len, do_sample=False)
            return final[0]["summary_text"]
        except Exception:
            return " ".join(parts)
    return parts[0]

def rag_recommended_places(destination, interests=None, limit=6):
    ints = sanitize_interests(interests)
    queries = [f"Top attractions in {destination}", f"Tourist attractions {destination}", f"{destination} sightseeing"]
    for i in ints[:3]:
        queries.append(f"{destination} {i}")

    titles = []
    for q in queries:
        for t in wiki_search_titles(q, limit=limit):
            if t not in titles: titles.append(t)
            if len(titles) >= limit*2: break
        if len(titles) >= limit*2: break

    results = []
    for t in titles[:limit]:
        text, url = wiki_fetch_text_for_title(t)
        if not text: continue
        results.append({"title": t, "snippet": bart_summarize(text), "url": url})
    return results

# ---- Agent ----
def decide_mode(days, interests, max_budget=None, mode_hint=None):
    if mode_hint in CANDIDATE_MODES:
        return mode_hint
    ints = ", ".join(sanitize_interests(interests)) or "general"
    prompt = f"Plan a {days}-day trip considering interests: {ints}. Budget: {max_budget if max_budget else 'unknown'}."
    try:
        res = mode_classifier(prompt, candidate_labels=CANDIDATE_MODES, multi_label=False)
        label = res.get("labels", [None])[0]
        score = res.get("scores", [0])[0]
        if label and score >= 0.45: return label
    except Exception:
        pass
    try:
        return "adventure" if int(days) <= 6 else "budget-friendly"
    except:
        return "budget-friendly"

# ---- Activity Pools (varied) ----
ACTIVITY_POOL = {
    "general": [
        ("City orientation walk", "morning", "walk", 0),
        ("Local market exploration", "afternoon", "culture", 5),
        ("Scenic viewpoint visit", "evening", "relax", 0),
        ("Park picnic", "afternoon", "relax", 10),
        ("River/harbor short cruise", "afternoon", "leisure", 20),
        ("Photography stroll", "morning", "experience", 0),
    ],
    "food": [
        ("Street-food tasting tour", "evening", "food", 12),
        ("Hands-on cooking class", "afternoon", "experience", 50),
        ("Local bakery & dessert crawl", "morning", "food", 8),
        ("Farmers' market visit", "morning", "food", 0),
        ("Chef's table dinner", "evening", "food", 70),
    ],
    "history": [
        ("Guided history tour", "morning", "museum", 18),
        ("Old town heritage walk", "afternoon", "walk", 0),
        ("Landmark museum visit", "afternoon", "museum", 15),
        ("Historic monument visit", "morning", "culture", 10),
        ("Archaeological site visit", "morning", "culture", 12),
    ],
    "adventure": [
        ("Hiking trail", "morning", "adventure", 25),
        ("Cycling countryside", "morning", "adventure", 15),
        ("Kayaking or boating", "afternoon", "adventure", 35),
        ("Rock-climbing session", "afternoon", "adventure", 40),
        ("Zipline or canopy tour", "afternoon", "adventure", 45),
    ],
    "luxury": [
        ("Spa & wellness session", "afternoon", "relax", 90),
        ("Fine dining reservation", "evening", "food", 120),
        ("Private guided tour", "morning", "experience", 110),
        ("Boutique shopping experience", "afternoon", "shopping", 150),
    ],
    "nightlife": [
        ("Rooftop bar", "evening", "nightlife", 30),
        ("Night market & music", "evening", "nightlife", 15),
        ("Live cultural performance", "night", "culture", 25),
    ]
}

def weighted_pool(interests, mode):
    ints = sanitize_interests(interests)
    keys = ["general"]
    for k in ["food", "history", "adventure", "luxury"]:
        if k in ints or (mode == k):
            keys.append(k)
    return list(dict.fromkeys(keys))

def pick_activities(keys, used, count, allow_night=False):
    picks = []
    cand = []
    for k in keys:
        for item in ACTIVITY_POOL.get(k, []):
            if k == "nightlife" and not allow_night: continue
            if item[0] not in used: cand.append(item)
    random.shuffle(cand)
    for i in cand:
        if len(picks) >= count: break
        picks.append(i); used.add(i[0])
    return picks

def generate_day_plan_varied(day_idx, rag_places, mode, interests, used):
    acts = []
    allow_night = "nightlife" in sanitize_interests(interests)

    # RAG place first
    if rag_places:
        featured = rag_places[day_idx % len(rag_places)]
        title = f"Visit: {featured['title']}"
        acts.append({"activity_id": new_id("act"), "title": title, "time_of_day": "morning", "type": "attraction", "estimated_cost": 0, "source": "RAG"})
        used.add(title)

    # pick 3 more
    keys = weighted_pool(interests, mode)
    if allow_night: keys.append("nightlife")

    picks = pick_activities(keys, used, 3, allow_night)
    # assign time slots
    slots = ["afternoon", "evening", "morning"]
    for idx, it in enumerate(picks):
        title, default_time, typ, cost = it
        tod = default_time if default_time in ["morning","afternoon","evening","night"] else slots[idx]
        acts.append({"activity_id": new_id("act"), "title": title, "time_of_day": tod, "type": typ, "estimated_cost": float(cost), "source": "gen"})

    return acts

# ---- Build Itinerary ----
def generate_itinerary(user_id, destination, days, interests=None, max_budget=None, mode_hint=None):
    days = int(days)
    mode = decide_mode(days, interests, max_budget, mode_hint)
    rag_places = rag_recommended_places(destination, interests, limit=min(8, 2 * days))

    used = set()
    day_plans = []
    for d in range(days):
        day_plans.append({"day": d+1, "date": None, "activities": generate_day_plan_varied(d, rag_places, mode, interests, used)})

    return {
        "itinerary_id": new_id("trip"),
        "user_id": user_id,
        "destination": destination,
        "days": days,
        "mode": mode,
        "interests": sanitize_interests(interests),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "day_plans": day_plans
    }

# ---- Budget (INR – Mid-Range India Travel) ----
def estimate_budget_inr(days, mode, num_people=2, flight_per_person=None):
    """
    Returns budget in INR ₹ with mid-range realistic pricing for India trips.
    Values are totals for the group. Nights = days - 1 (min 1).
    """
    days = int(days)

    # Per-person, per-day INR baselines
    lodgings = {
        "budget-friendly": 1200,
        "adventure": 2000,
        "luxury": 6500
    }
    meals = {
        "budget-friendly": 500,
        "adventure": 800,
        "luxury": 2200
    }
    activities = {
        "budget-friendly": 400,
        "adventure": 1200,
        "luxury": 2800
    }

    stay_cost = lodgings.get(mode, 1200) * num_people * max(1, days - 1)   # nights
    food_cost = meals.get(mode, 500) * num_people * days
    act_cost = activities.get(mode, 400) * num_people * days
    flight_cost = (float(flight_per_person) * num_people) if flight_per_person else 0.0

    total = stay_cost + food_cost + act_cost + flight_cost

    return {
        "currency": "INR",
        "breakdown": {
            "lodging": f"₹ {stay_cost:,}",
            "meals": f"₹ {food_cost:,}",
            "activities": f"₹ {act_cost:,}",
            "flights": f"₹ {int(flight_cost):,}" if flight_cost else "₹ 0"
        },
        "total_estimate": f"₹ {int(total):,}"
    }

# ---- Endpoints ----
@app.route("/plan_trip", methods=["POST"])
def plan_trip():
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("user_id"); dest = data.get("destination"); days = data.get("days")
    if not user_id or not dest or not days:
        return err("Fields required: user_id, destination, days")
    ensure_user(user_id)
    trip = generate_itinerary(user_id, dest, days, data.get("interests"), data.get("max_budget"), data.get("mode"))
    USERS[user_id]["itinerary"] = trip
    USERS[user_id]["activities"] = {a["activity_id"]:a for dp in trip["day_plans"] for a in dp["activities"]}
    return jsonify({"ok": True, "itinerary": trip})

@app.route("/get_itinerary/<user_id>", methods=["GET"])
def get_itinerary(user_id):
    ensure_user(user_id)
    it = USERS[user_id]["itinerary"]
    if not it: return err("No itinerary found.", 404)
    return jsonify({"ok": True, "itinerary": it})

@app.route("/update_day_plan", methods=["POST"])
def update_day():
    data = request.get_json(force=True, silent=True) or {}
    user_id, day, acts_in = data.get("user_id"), data.get("day"), data.get("activities")
    if not user_id or not day or not isinstance(acts_in,list):
        return err("Fields: user_id, day, activities[]")
    ensure_user(user_id)
    it = USERS[user_id]["itinerary"]
    if not it: return err("No itinerary found.",404)
    di = int(day)-1
    if di<0 or di>=len(it["day_plans"]): return err("Invalid day")
    new = []
    for a in acts_in:
        act = {
            "activity_id": new_id("act"),
            "title": a.get("title","Custom Activity"),
            "time_of_day": a.get("time_of_day","any"),
            "type": a.get("type","custom"),
            "estimated_cost": float(a.get("estimated_cost",0)),
            "source":"user"
        }
        new.append(act); USERS[user_id]["activities"][act["activity_id"]] = act
    it["day_plans"][di]["activities"] = new
    return jsonify({"ok":True,"day_plan":it["day_plans"][di]})

@app.route("/add_activity", methods=["POST"])
def add_activity():
    data = request.get_json(force=True, silent=True) or {}
    user_id, day, a = data.get("user_id"), data.get("day"), data.get("activity")
    if not user_id or not day or not isinstance(a,dict):
        return err("Fields: user_id, day, activity{}")
    ensure_user(user_id)
    it = USERS[user_id]["itinerary"]
    if not it: return err("No itinerary found.",404)
    di = int(day)-1
    if di<0 or di>=len(it["day_plans"]): return err("Invalid day")
    act = {
        "activity_id": new_id("act"),
        "title": a.get("title","Custom Activity"),
        "time_of_day": a.get("time_of_day","any"),
        "type": a.get("type","custom"),
        "estimated_cost": float(a.get("estimated_cost",0)),
        "source":"user"
    }
    it["day_plans"][di]["activities"].append(act)
    USERS[user_id]["activities"][act["activity_id"]] = act
    return jsonify({"ok":True,"activity":act,"day_plan":it["day_plans"][di]})

@app.route("/remove_activity/<activity_id>", methods=["DELETE"])
def remove_activity(activity_id):
    user_id = request.args.get("user_id") or (request.get_json(silent=True) or {}).get("user_id")
    if not user_id: return err("user_id required")
    ensure_user(user_id)
    it = USERS[user_id]["itinerary"]
    if not it: return err("No itinerary found.",404)
    found=False
    for dp in it["day_plans"]:
        keep=[]
        for a in dp["activities"]:
            if a["activity_id"]==activity_id: found=True; continue
            keep.append(a)
        dp["activities"]=keep
    USERS[user_id]["activities"].pop(activity_id, None)
    if not found: return err("Activity not found.",404)
    return jsonify({"ok":True,"removed_activity_id":activity_id})

@app.route("/recommended_places", methods=["GET"])
def recommended():
    dest = request.args.get("destination")
    ints = request.args.get("interests")
    if not dest: return err("destination required")
    return jsonify({"ok":True,"destination":dest,"results":rag_recommended_places(dest,ints)})

@app.route("/share_itinerary", methods=["POST"])
def share():
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("user_id")
    if not user_id: return err("user_id required")
    ensure_user(user_id)
    it = USERS[user_id]["itinerary"]
    if not it: return err("No itinerary found.",404)

    dest, days, mode = it['destination'], it['days'], it['mode']
    interests = ", ".join(it["interests"]) if it["interests"] else "general interests"

    lines=[]
    lines.append(f"**Your {days}-Day {mode.title()} Trip to {dest}**\n")
    lines.append(f"This trip is tailored for someone interested in **{interests}**. Here's what your experience looks like:\n")

    for dp in it["day_plans"]:
        lines.append(f"**Day {dp['day']}**")
        for a in dp["activities"]:
            when = a.get("time_of_day","")
            cost = f" (~{a['estimated_cost']})" if a.get("estimated_cost") else ""
            lines.append(f"- {when.capitalize()+': ' if when else ''}{a['title']}{cost}")
        lines.append("")

    lines.append("Enjoy your trip! Safe travels and unforgettable memories await!")

    return jsonify({"ok":True,"summary_text":"\n".join(lines)})

@app.route("/budget_estimate", methods=["POST"])
def budget():
    data = request.get_json(force=True, silent=True) or {}
    days = data.get("days")
    if not days: return err("days required")
    mode = data.get("mode")
    if mode not in CANDIDATE_MODES:
        mode = decide_mode(days, data.get("interests"), data.get("max_budget"), None)
    est = estimate_budget_inr(days, mode, int(data.get("num_people",2)), data.get("flight_per_person"))
    return jsonify({"ok":True,"mode":mode,"estimate":est})

@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok":True,"service":"Travel Planner API (BART)","version":"0.5-inr"})

if __name__ == "__main__":
    random.seed()
    app.run(debug=True, port=5000)
