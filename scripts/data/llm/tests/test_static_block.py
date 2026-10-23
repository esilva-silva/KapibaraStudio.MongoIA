"""Bloqueio estatico de escrita/ddl/execucao arbitraria (common.static_check) e escaneamento JS."""
import pytest

from common import static_check

BAD = [
    "db.users.insertOne({a: 1})",
    "db.users.insertMany([{a:1}])",
    "db.users.updateOne({a:1},{$set:{b:2}})",
    "db.users.updateMany({}, {$inc:{n:1}})",
    "db.users.deleteMany({})",
    "db.users.remove({})",
    "db.users.drop()",
    "db.dropDatabase()",
    "db.users.createIndex({a:1})",
    "db.users.bulkWrite([])",
    "db.users.findOneAndUpdate({}, {$set:{a:1}})",
    "db.users.findAndModify({query:{}, update:{}})",
    "db.users.replaceOne({}, {})",
    "db.users.aggregate([{$match:{}},{$out:'x'}])",
    "db.users.aggregate([{$merge:{into:'x'}}])",
    "db.users.aggregate([{$addFields:{x:{$function:{body:'function(){}',args:[],lang:'js'}}}}])",
    "db.users.find({$where:'this.a==1'})",
    "eval('1+1')",
    "load('/etc/passwd')",
    "require('fs').readFileSync('/etc/passwd')",
    "db.runCommand({insert:'x', documents:[]})",
    "db.adminCommand({shutdown:1})",
    "db.users.renameCollection('b')",
    "db.createCollection('z')",
    "while(true) {}",
    "db.users.find().maxTimeMS(999999)",
    "process.exit(1)",
    "use otherdb",
]
GOOD = [
    "db.users.find({ role: 'admin' }).sort({ name: 1 }).limit(5)",
    "db.orders.aggregate([{ $match: { status: 'open' } }, { $group: { _id: '$region', n: { $sum: 1 } } }]).toArray()",
    "const start = new Date('2025-01-01');\ndb.orders.countDocuments({ placedAt: { $gte: start } })",
    "db.getSiblingDB('tenantly').getCollection('users').find({ update: 1 })",   # `update` como nome de campo (chave)
    "db.logs.find({ message: /insert into/i })",                               # 'insert' dentro de regex literal
    "db.logs.find({ message: 'please drop the table' })",                      # dentro de string
    "// insertOne is not used here\ndb.logs.find()",                            # em comentario
    "db.t.aggregate([{ $facet: { a: [{ $count: 'n' }] } }], { allowDiskUse: true })",
]


@pytest.mark.parametrize("text", BAD)
def test_bad_is_blocked(text):
    assert static_check(text, {"tenantly"}), text


@pytest.mark.parametrize("text", GOOD)
def test_good_passes(text):
    assert static_check(text, {"tenantly"}) == [], static_check(text, {"tenantly"})


def test_getsiblingdb_other_db_blocked():
    assert any("getSiblingDB" in r for r in static_check("db.getSiblingDB('admin').users.find()", {"tenantly"}))
    assert any("getSiblingDB" in r for r in static_check("db.getSiblingDB(name).users.find()", {"tenantly"}))
    assert static_check("db.getSiblingDB('tenantly').users.find()", {"tenantly"}) == []
