import jsscan

T = '''const lim = 5;
const pipeline = [
  { $match: { status: "open", 'a.b': { $gte: new Date("2025-01-01") } } },
  { $group: { _id: "$region", total: { $sum: "$amount" } } }
];
db.getCollection("orders").aggregate(pipeline).toArray();
db.orders.find({ qty: { $gt: 3 } }).limit(lim)
'''


def roles(text):
    sc = jsscan.scan(text)
    return {(r, text[s:e]) for s, e, r, _ in sc["spans"]}


def test_spans_basic():
    sp = roles(T)
    assert ("key", "status") in sp and ("key", "a.b") in sp
    assert ("op", "$match") in sp and ("op", "$gte") in sp
    assert ("fieldref", "$region") in sp
    assert ("value", 'new Date("2025-01-01")') in sp
    assert ("strarg", "orders") in sp
    assert any(r == "stage" and t.startswith("{ $match") for r, t in sp)
    assert ("member", "orders") in sp
    assert ("method", "aggregate") in sp
    assert sum(1 for r, _ in sp if r == "stmt") >= 4


def test_mask_hides_strings_and_comments():
    m = jsscan.mask('db.x.find({a: "insertOne"}) // drop()')
    assert "insertOne" not in m and "drop" not in m and m.startswith("db.x.find")


def test_regex_and_template():
    sc = jsscan.scan(r"db.x.find({ n: /a\/b/i, t: `x${1}y` })")
    kinds = [t[0] for t in sc["tokens"]]
    assert "regex" in kinds and "tpl" in kinds
