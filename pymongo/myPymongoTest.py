import pymongo
import datetime
import pprint
from bson.objectid import ObjectId

client = pymongo.MongoClient()
dbs = ['mydb0','mydb1','mydb2'] # 要创建的数据库
cols = 'mycoll'  # 要创建的集合
post = {"author": "Mike",
		"text": "My first blog post!",
		"tags": ["mongodb", "python", "pymongo"],
		"date": datetime.datetime.utcnow()
		}
post1 = {"author": "Mike",
		"text": "My first blog post!",
		"tags": ["mongodb", "python", "pymongo"],
		"date": datetime.datetime.utcnow()
		}
# 循环创建多个数据库
# for i in dbs:
# 	db = client[i]
# 	collection = db[cols]
#开始创建数据库,获取数据库对象
# db = client['mydb0'] #获取数据库对象方法1
db = client.mydb0 #获取数据库对象方法2
db.drop_collection('mycol9') # 删除集合
#开始创建集合
collection = db.mycol #获取集合对象方法1
collection = db['mycol9'] #获取集合对象方法2
# 使用insert_one()方法插入一条文档,并返回插入文档的ObjectId,返回值是一个ObjectId对象.
post_id = collection.insert_one(post).inserted_id
# 使用save()方法插入一条文档,功能类似insert,但如果ObjectId已存在,则会替换文档.所以无法插入多条存在文档.
rest = collection.save(post)
rest = collection.save(post) # save()方法同一变量可以多次插入不会报错,但是由于ID相同,只会插入一条数据.
result = collection.insert_many([{"x":i} for i in range(3)])
print(post_id) #ObjectId(5bff4d4b738ab40ffc797fab)
pprint.pprint(collection.find_one()) # 使用pprint()格式化输出
print(collection.find_one())
# 列出所有集合名
print(db.collection_names(include_system_collections=True))
post_id = collection.insert_one(post1).inserted_id# #post1改成post则会报错无法插入,同一变量多次插入,pymongo认为是重复插入.
print(post_id) #5bffb782738ab415b866130e
post_id_as_str = str(post_id) #
print(db.collection.find_one({"_id":post_id_as_str}))
def get(post_id):
	print(post_id)
	document = collection.find_one({"_id":ObjectId(post_id)}) #通过ObjectId对象转id值获取文档
	pprint.pprint(document)

get(post_id)

new_posts = [{
"author": "Mike",
"text": "Another post!",
"tags": ["bulk", "insert"],
"date": datetime.datetime(2009, 11, 12, 11, 14)
},
{
"author": "Eliot",
"title": "MongoDB is fun",
"text": "and pretty easy too!",
"date": datetime.datetime(2009, 11, 10, 10, 45)
}
]
# 一次性插入多个文档,并返回[ObjectId,ObjectId]对象列表
results = collection.insert_many(new_posts)
print(results.inserted_ids) #[ObjectId('5bffd186738ab442f0005539'), ObjectId('5bffd186738ab442f000553a')]

#循环打印find()查找到的文档信息
for inst in collection.find({'author':'Mike'}):
	pprint.pprint(inst)
#count_documents() 统计文档个数
print(collection.count_documents({'author':'Mike'}))
# # (self, year: int, month: int = ..., day: int = ..., hour: int = ...,minute: int = ..., second: int = ..., microsecond: int = ...,tzinfo: Optional[tzinfo] = ...)
#使用datetime()转为pymongo的ISODate()格式,并进行时间计算.
d = datetime.datetime(2018,11,25,15,2,10)
for i in collection.find({"date":{"$lt":d}}).sort("author"):
	pprint.pprint(i)
#创建升序索引pymongo.ASCENDING=1 pymongo.DESCENDING=-1
result = collection.create_index([("author",pymongo.ASCENDING)],unique=False)
result = collection.index_information()# 获取索引的详细信息
pprint.pprint(result)
print(sorted(list(collection.index_information()))) # 列出已有的索引,只有索引域['_id_', 'author_1']
requests = [pymongo.InsertOne({'y': 1}), pymongo.DeleteOne({'y': 1}),pymongo.ReplaceOne({'x': 1}, {'z': 1}, upsert=True)]
#pymongo支持批量请求处理操作bulk_write(requests,ordered)如果order=True表示按顺序处理.
result = collection.bulk_write(requests,ordered=True)
print(result.inserted_count)
print(result.deleted_count)
print(result.modified_count)
print(result.upserted_ids)

client.close()

"""
5bffd561738ab426e8aaa601
{'_id': ObjectId('5bffd561738ab426e8aaa601'),
 'author': 'Mike',
 'date': datetime.datetime(2018, 11, 29, 12, 2, 41, 108000),
 'tags': ['mongodb', 'python', 'pymongo'],
 'text': 'My first blog post!'}
{'_id': ObjectId('5bffd561738ab426e8aaa601'), 'author': 'Mike', 'text': 'My first blog post!', 'tags': ['mongodb', 'python', 'pymongo'], 'date': datetime.datetime(2018, 11, 29, 12, 2, 41, 108000)}
['mycol9']
5bffd561738ab426e8aaa605
None
5bffd561738ab426e8aaa605
{'_id': ObjectId('5bffd561738ab426e8aaa605'),
 'author': 'Mike',
 'date': datetime.datetime(2018, 11, 29, 12, 2, 41, 108000),
 'tags': ['mongodb', 'python', 'pymongo'],
 'text': 'My first blog post!'}
[ObjectId('5bffd561738ab426e8aaa606'), ObjectId('5bffd561738ab426e8aaa607')]
{'_id': ObjectId('5bffd561738ab426e8aaa601'),
 'author': 'Mike',
 'date': datetime.datetime(2018, 11, 29, 12, 2, 41, 108000),
 'tags': ['mongodb', 'python', 'pymongo'],
 'text': 'My first blog post!'}
{'_id': ObjectId('5bffd561738ab426e8aaa605'),
 'author': 'Mike',
 'date': datetime.datetime(2018, 11, 29, 12, 2, 41, 108000),
 'tags': ['mongodb', 'python', 'pymongo'],
 'text': 'My first blog post!'}
{'_id': ObjectId('5bffd561738ab426e8aaa606'),
 'author': 'Mike',
 'date': datetime.datetime(2009, 11, 12, 11, 14),
 'tags': ['bulk', 'insert'],
 'text': 'Another post!'}
3
{'_id': ObjectId('5bffd561738ab426e8aaa607'),
 'author': 'Eliot',
 'date': datetime.datetime(2009, 11, 10, 10, 45),
 'text': 'and pretty easy too!',
 'title': 'MongoDB is fun'}
{'_id': ObjectId('5bffd561738ab426e8aaa606'),
 'author': 'Mike',
 'date': datetime.datetime(2009, 11, 12, 11, 14),
 'tags': ['bulk', 'insert'],
 'text': 'Another post!'}
{'_id_': {'key': [('_id', 1)], 'ns': 'mydb0.mycol9', 'v': 2},
 'author_1': {'key': [('author', 1)], 'ns': 'mydb0.mycol9', 'v': 2}}
['_id_', 'author_1']
1
1
1
{}
"""
