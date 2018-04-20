from bson.objectid import ObjectId
import json
import pymongo
from pymongo import MongoClient
import sys

fish_client = MongoClient('mongodb://localhost:27017')
fish_db = fish_client.fish_database
fish_posts = fish_db.posts
 
def db_save(post):
    inserted_id = fish_posts.insert_one(post).inserted_id
    return str(inserted_id)

def db_load(post_id):
    post = fish_posts.find_one({'_id': ObjectId(post_id)})
    post['id'] = str(post['_id'])
    del post['_id']
    return post

if __name__ == '__main__':
    reader = open(sys.argv[1], 'r')
    param = json.load(reader)
    reader.close()
    post_id = db_save(param)

    post = db_load(post_id)
    print json.dumps(post, indent = 4)

