import json
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch as t
import xgboost as xgb
from gensim import corpora, models, similarities
from py2neo import Graph, NodeMatcher, Subgraph
from py2neo import Node, Relationship
from sklearn import svm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from openke.config import Trainer, Tester
from openke.data import TrainDataLoader, TestDataLoader
from openke.module.loss import MarginLoss
from openke.module.model import TransD
from openke.module.strategy import NegativeSampling


def get_statistics():
    """
    每个人的平均帖子数量，最大最小帖子数量
    平均帖子长度，最大最小帖子长度
    """

    '''
    mypersonality
    平均帖子数量：155 最大：2441 最小：1
    平均帖子长度：15.3 最大 1001 最小：1
    '''
    data = pd.read_csv('dataset/youtube_for_kg.csv')
    data = data.posts.values
    avg_user_posts_num, max_user_posts_num, min_user_posts_num, total_user_posts_num = 0, 0, 10000, 0
    avg_post_length, max_post_length, min_post_length, total_post_length = 0, 0, 10000, 0

    for i in tqdm(range(data.shape[0])):
        posts = data[i].split('<sep>')
        posts = [p.strip() for p in posts if len(p.strip()) != 0]
        user_posts_num = len(posts)
        total_user_posts_num += user_posts_num
        max_user_posts_num = max(max_user_posts_num, user_posts_num)
        min_user_posts_num = min(min_user_posts_num, user_posts_num)
        post_length = [len(p.strip().split()) for p in posts if len(p.strip()) != 0]
        if len(post_length) == 0:
            continue
        total_post_length += sum(post_length)
        max_post_length = max(max(post_length), max_post_length)
        min_post_length = min(min(post_length), min_post_length)

    avg_user_posts_num = total_user_posts_num / data.shape[0]
    avg_post_length = total_post_length / total_user_posts_num

    print('帖子总数量为：{:}\n'.format(total_user_posts_num))
    print('用户的平均帖子数量为：{:}\n'.format(avg_user_posts_num))
    print('用户的最大帖子数量为：{:}\n'.format(max_user_posts_num))
    print('用户的最小帖子数量为：{:}\n'.format(min_user_posts_num))
    print('用户的平均帖子长度为：{:}\n'.format(avg_post_length))
    print('用户的最大帖子长度为：{:}\n'.format(max_post_length))
    print('用户的最小帖子长度为：{:}\n'.format(min_post_length))


def insertIdIntoCsv(dataset):
    if dataset == 'youtube':
        path = '/home/gzm/personality/EMNLP/dataset/youtube_for_kg.csv'
    elif dataset == 'pan':
        path = '/home/gzm/personality/EMNLP/dataset/pan_for_kg.csv'
    elif dataset == 'mypersonality':
        path = '/home/gzm/personality/EMNLP/dataset/my_for_kg.csv'

    data = pd.read_csv(path, index_col=0)
    data_name = ['VLOG{}'.format(i + 1) for i in range(data.shape[0])]
    data.insert(0, 'id', data_name)
    data.to_csv(path)


def show_personality_destribution(dataset='youtube'):
    """
    显示数据集中，不同人格标签的评分分布情况
    :param dataset:
    :return:
    """
    if dataset == 'youtube':
        data = pd.read_csv('/home/gzm/personality/dataset/youtube_liwc.csv')
    elif dataset == 'pan':
        data = pd.read_csv('/home/gzm/personality/dataset/youtube_liwc.csv')
    elif dataset == 'mypersonality':
        data = pd.read_csv('/home/gzm/personality/dataset/youtube_liwc.csv')
    ope, con, ext, agr, neu = {}, {}, {}, {}, {}
    for i in range(1, 10):
        ope['{}%'.format(i * 10)] = data.ope.quantile(0.1 * i)
        con['{}%'.format(i * 10)] = data.con.quantile(0.1 * i)
        ext['{}%'.format(i * 10)] = data.ext.quantile(0.1 * i)
        agr['{}%'.format(i * 10)] = data.agr.quantile(0.1 * i)
        neu['{}%'.format(i * 10)] = data.neu.quantile(0.1 * i)
    print('ope distribution: ', ope)
    print('con distribution: ', con)
    print('ext distribution: ', ext)
    print('agr distribution: ', agr)
    print('neu distribution: ', neu)


def generate_user_feature(dataset='youtube'):
    """
    从数据集中读取数据，并根据特征词使用情况构造特征词典和标签
    :param dataset: *_result_simp.csv
    :return: user_language_feature_dict
    """
    with open('dataset/entity.pkl', 'rb') as f:
        entity = pickle.load(f)
    if dataset == 'youtube':
        user_language_feature = pd.read_csv('../datasets/youtube_for_kg.csv')
    elif dataset == 'pan':
        user_language_feature = pd.read_csv('../datasets/pan_for_kg.csv')
    elif dataset == 'my':
        user_language_feature = pd.read_csv('../datasets/my_for_kg.csv')
        user_language_feature = user_language_feature[:12000]
    features = list(user_language_feature.columns)
    features = features[8: -10]
    print(features)
    print(entity['word'])
    word_list = []
    with open('dataset/wordlist.json', 'r') as f:
        word_list = json.load(f)
    print(word_list)

    user_language_feature_dict = {}

    for i in range(user_language_feature.shape[0]):
        user_id = user_language_feature.loc[i, 'id']
        user_language_feature_dict[user_id] = []
        for feature in features:
            flag = int(user_language_feature.loc[i, feature])
            if flag > 0 and (feature + '+') in entity['word']:
                user_language_feature_dict[user_id].append(feature + '+')
            if flag < 0 and (feature + '-') in entity['word']:
                user_language_feature_dict[user_id].append(feature + '-')
            # if flag > 0 and (feature + '+') in word_list:
            #     user_language_feature_dict[user_id].append(feature + '+')
            # if flag < 0 and (feature + '-') in word_list:
            #     user_language_feature_dict[user_id].append(feature + '-')

    return user_language_feature_dict


# 老版本，弃用
def add_user_into_kg(dataset='youtube', train_rate=0.5, split_rate=0.2, version=1):
    """
    根据用户特征字典，将数据集用户自动添加至知识图谱
    每次添加前需手动删除上一次添加的用户
    第一个循环的代码为添加用户，如果用户已经添加，则需要注释掉
    :param dataset:
    :param user_feature_dict:
    :param rate:两侧保留的比例
    :return:
    """
    graph = Graph('http://10.112.49.188:7477', username='neo4j', password='962464')
    if dataset == 'youtube':
        node_person_type = 'YoutubeUser'
        user_feature_dict = generate_user_feature(dataset)
        data = pd.read_csv('/home/gzm/personality/dataset/youtube_liwc.csv')
    elif dataset == 'pan':
        node_person_type = 'PanUser'
        user_feature_dict = generate_user_feature(dataset)
        data = pd.read_csv('/home/gzm/personality/dataset/youtube_liwc.csv')
    elif dataset == 'mypersonality':
        node_person_type = 'MyPersonalityUser'
        user_feature_dict = generate_user_feature(dataset)
        data = pd.read_csv('/home/gzm/personality/dataset/youtube_liwc.csv')
    matcher = NodeMatcher(graph)
    # for k, v in user_feature_dict.items():
    #     node_person = Node(node_person_type, name=k)
    #     graph.create(node_person)
    #     for feature in v:
    #         node_feature = matcher.match('word', name=feature).first()
    #         r = Relationship(node_person, 'USES', node_feature)
    #         graph.create(r)
    #         print('添加关系（{}）- [{}] - （{}）'.format(node_person['name'], 'USES', node_feature['name']))

    train_user_list = []

    ope_split_low, ope_split_high = data.ope.quantile(split_rate), data.ope.quantile(1 - split_rate)
    con_split_low, con_split_high = data.con.quantile(split_rate), data.con.quantile(1 - split_rate)
    ext_split_low, ext_split_high = data.ext.quantile(split_rate), data.ext.quantile(1 - split_rate)
    agr_split_low, agr_split_high = data.agr.quantile(split_rate), data.agr.quantile(1 - split_rate)
    neu_split_low, neu_split_high = data.neu.quantile(split_rate), data.neu.quantile(1 - split_rate)
    data = shuffle(data)
    for i in range(data.shape[0]):
        if i > int(data.shape[0] * train_rate):
            break
        labels = data.loc[i, ['ope', 'con', 'agr', 'ext', 'neu']].values
        node_person = matcher.match(node_person_type, name=data.loc[i, 'id']).first()
        for label, label_name, value_split in zip(labels, ['ope', 'con', 'agr', 'ext', 'neu'],
                                                  [(ope_split_low, ope_split_high),
                                                   (con_split_low, con_split_high),
                                                   (agr_split_low, agr_split_high),
                                                   (ext_split_low, ext_split_high),
                                                   (neu_split_low, neu_split_high)]):
            if label <= value_split[0]:
                node_personality_name = label_name + '-low'
            elif label > value_split[0] and label < value_split[1]:
                node_personality_name = label_name + '-avg'
            else:
                node_personality_name = label_name + '-high'
            node_personality = matcher.match('personality', name=node_personality_name).first()
            r = Relationship(node_person, label_name + 'Is', node_personality)
            graph.create(r)
            print('添加关系（{}）- [{}] -（{})'.format(node_person['name'], label_name + 'Is', node_personality['name']))
        train_user_list.append(data.loc[i, 'id'])
    print('DONE')
    '''
    train_user_dataset.json用于保存训练集名单，命名格式需要调整
    trainlist_dataset_trainrate_splitrate_version.json
    '''
    with open('dataset/trainlist_{}_{}_{}_{}.json'.format(dataset, train_rate, split_rate, version), 'w') as f:
        json.dump(train_user_list, f)
    return train_user_list


# 添加知识图谱的语言风格部分
def add_user_linguistic(dataset='youtube'):
    """
    将指定数据集用户按照语言风格加入图谱，仅构造与语言风格相关的部分
    通过批量提交，将时间缩短为三分之一
    :param dataset:
    :return:
    """
    graph = Graph('http://10.112.170.242:7477', username='neo4j', password='962464')
    if dataset == 'youtube':
        node_person_type = 'YoutubeUser'
        user_feature_dict = generate_user_feature(dataset)
        print('已生成user_feature_dict')
    elif dataset == 'pan':
        node_person_type = 'PanUser'
        user_feature_dict = generate_user_feature(dataset)
        print('已生成user_feature_dict')
    elif dataset == 'my':
        node_person_type = 'MyUser'
        user_feature_dict = generate_user_feature(dataset)
        print('已生成user_feature_dict')
    matcher = NodeMatcher(graph)

    start = time.time()
    tx = graph.begin()
    nodes_list = []
    print('开始添加节点')
    for k, v in tqdm(user_feature_dict.items()):
        nodes_list.append(Node(node_person_type, name=k))
    nodes_list = Subgraph(nodes_list)
    tx.create(nodes_list)
    tx.commit()

    mid = time.time()
    relation_list = []
    print('开始添加关系')
    count = 0
    for k, v in tqdm(user_feature_dict.items()):
        for feature in v:
            node_person = matcher.match(node_person_type, name=k).first()
            node_feature = matcher.match('word', name=feature).first()
            relation_list.append(Relationship(node_person, 'USES', node_feature))
            # graph.run('match (m:{} {{name:\'{}\'}}), (n:word {{name:\'{}\'}})   '
            #                 'create (m) - [r:USES] -> (n)'.format(node_person_type, k, feature))
            # print('创建关系（{}）- [{}] - （{}）'.format(k, 'USES', feature))
        count += 1
        if count % 1000 == 0:
            relation = Subgraph(relationships=relation_list)
            graph.create(relation)
            relation_list = []
            print('已经提交{}个用户关系'.format(count))
    if len(relation_list) != 0:
        relation_list = Subgraph(relationships=relation_list)
        graph.create(relation_list)
    end = time.time()
    print('批量提交节点耗时{} cypher创建关系耗时{} '.format(mid - start, end - mid))

    # start = time.time()
    # for k, v in user_feature_dict.items():
    #     node_person = Node(node_person_type, name=k)
    #     graph.create(node_person)
    #     for feature in v:
    #         node_feature = matcher.match('word', name=feature).first()
    #         r = Relationship(node_person, 'USES', node_feature)
    #         graph.create(r)
    #         print('添加关系（{}）- [{}] - （{}）'.format(node_person['name'], 'USES', node_feature['name']))
    # end = time.time()
    # print('原始方法耗时 {} '.format(end - start))


# 添加知识图谱的人格部分
def add_user_personality(dataset='youtube', train_rate=0.5, split_rate=0.2, version=1):
    """
    不添加节点，仅构造用户与人格节点之间的关系，使用前要确保用户节点已经添加至图谱中
    :param dataset:
    :param train_rate:
    :param split_rate:
    :param version:
    :return:保存训练集名单 trainlist_dataset_trainrate_splitrate_version.json

    """
    graph = Graph('http://10.112.170.242:7477', username='neo4j', password='962464')
    if dataset == 'youtube':
        node_person_type = 'YoutubeUser'
        data = pd.read_csv('../datasets/youtube_for_kg.csv')
    elif dataset == 'pan':
        node_person_type = 'PanUser'
        data = pd.read_csv('../datasets/pan_for_kg.csv')
    elif dataset == 'my':
        node_person_type = 'MyUser'
        data = pd.read_csv('../datasets/my_for_kg.csv')
        data = data[:12000]
    matcher = NodeMatcher(graph)

    train_user_list = []
    relation_list = []

    ope_split_low, ope_split_high = data.ope.quantile(split_rate), data.ope.quantile(1 - split_rate)
    con_split_low, con_split_high = data.con.quantile(split_rate), data.con.quantile(1 - split_rate)
    ext_split_low, ext_split_high = data.ext.quantile(split_rate), data.ext.quantile(1 - split_rate)
    agr_split_low, agr_split_high = data.agr.quantile(split_rate), data.agr.quantile(1 - split_rate)
    neu_split_low, neu_split_high = data.neu.quantile(split_rate), data.neu.quantile(1 - split_rate)
    # data = shuffle(data)
    count = 0
    for i in tqdm(range(int(train_rate * data.shape[0]))):
        labels = data.loc[i, ['ope', 'con', 'agr', 'ext', 'neu']].values
        node_person = matcher.match(node_person_type, name=data.loc[i, 'id']).first()
        for label, label_name, value_split in zip(labels, ['ope', 'con', 'agr', 'ext', 'neu'],
                                                  [(ope_split_low, ope_split_high),
                                                   (con_split_low, con_split_high),
                                                   (agr_split_low, agr_split_high),
                                                   (ext_split_low, ext_split_high),
                                                   (neu_split_low, neu_split_high)]):
            if label <= value_split[0]:
                node_personality_name = label_name + '-low'
            elif label > value_split[0] and label < value_split[1]:
                node_personality_name = label_name + '-avg'
            else:
                node_personality_name = label_name + '-high'
            node_personality = matcher.match('personality', name=node_personality_name).first()
            r = Relationship(node_person, label_name + 'Is', node_personality)
            # print(r)
            relation_list.append(r)
            count += 1
            if count % 1000 == 0:
                relation = Subgraph(relationships=relation_list)
                graph.create(relation)
                relation_list = []
                print('已经提交{}个用户人格关系'.format(count))
            # graph.create(r)
            # print('添加关系（{}）- [{}] -（{})'.format(node_person['name'], label_name + 'Is', node_personality['name']))
        train_user_list.append(data.loc[i, 'id'])
    if len(relation_list) != 0:
        relation_list = Subgraph(relationships=relation_list)
        graph.create(relation_list)
    print('DONE')
    '''
    train_user_dataset.json用于保存训练集名单，命名格式需要调整
    trainlist_dataset_trainrate_splitrate_version.json
    '''
    with open('dataset/trainlist_{}_{}_{}_{}.json'.format(dataset, train_rate, split_rate, version), 'w') as f:
        json.dump(train_user_list, f)
    return train_user_list


# 使用tf-idf计算相似度，并在为相似度超过阈值的用户之间构建一条边
def add_user_similarity(dataset='youtube', threshhold=0.01):
    """
    不添加节点，仅构造用户与人格节点之间的关系，使用前要确保用户节点已经添加至图谱中
    :param dataset:
    :param train_rate:
    :param split_rate:
    :param version:
    :return:保存训练集名单 trainlist_dataset_trainrate_splitrate_version.json

    """
    graph = Graph('http://10.112.49.188:7477', username='neo4j', password='962464')
    split_flag = '<sep>'
    if dataset == 'youtube':
        node_person_type = 'YoutubeUser'
        data = pd.read_csv('../datasets/youtube_for_kg.csv')
    elif dataset == 'pan':
        node_person_type = 'PanUser'
        data = pd.read_csv('../datasets/pan_for_kg.csv')
    elif dataset == 'my':
        node_person_type = 'MyUser'
        data = pd.read_csv('../datasets/my_for_kg.csv')
        split_flag = '<SEP>'
        data = data[:12000]

    matcher = NodeMatcher(graph)

    all_doc = []
    all_doc_list = []

    posts = data.posts.values

    for i in range(posts.shape[0]):
        post = posts[i].replace(split_flag, ' ')
        all_doc.append(post)

    for doc in all_doc:
        doc_list = [word.lower() for word in doc.split()]
        all_doc_list.append(doc_list)

    dictionary = corpora.Dictionary(all_doc_list)
    corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]

    tfidf = models.TfidfModel(corpus)

    similarMatrix = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))

    relation_list = []
    count = 0
    for i in tqdm(range(data.shape[0])):
        user = data.loc[i, 'id']
        node_user = matcher.match(node_person_type, name=user).first()
        doc = dictionary.doc2bow(all_doc_list[i])
        sim = similarMatrix[tfidf[doc]]
        for j, s in enumerate(sim):
            if i != j and s > threshhold:
                count += 1
                user_compared = data.loc[j, 'id']
                node_user_compared = matcher.match(node_person_type, name=user_compared).first()
                properties = {'text similarity': float(s)}
                r = Relationship(node_user, 'textSimilarTo', node_user_compared, **properties)
                relation_list.append(r)
        if len(relation_list) == 0:
            continue
        relation = Subgraph(relationships=relation_list)
        graph.create(relation)
        relation_list = []
    print('Done!!')


# TODO
# 使用sentence_bert计算相似度，并在为相似度超过阈值的用户之间构建一条边
def add_user_similarity_v2(dataset='youtube', threshhold=0.6):
    """
    :param dataset:训练数据集
    :param threshhold:相似度阈值
    :return:构建好的图谱相似度关系子图
    """
    graph = Graph('http://10.112.49.188:7477', username='neo4j', password='962464')
    if dataset == 'youtube':
        node_person_type = 'YoutubeUser'
        data = pd.read_csv('../datasets/youtube_for_sim.csv')
    elif dataset == 'pan':
        node_person_type = 'PanUser'
        data = pd.read_csv('../datasets/pan_for_sim.csv')
    elif dataset == 'my':
        node_person_type = 'MyUser'
        data = pd.read_csv('../datasets/my_for_sim.csv')
        data = data[:12000]

    matcher = NodeMatcher(graph)
    relation_list = []
    count = 0
    for i in tqdm(range(data.shape[0])):
        user = data.loc[i, 'id']
        node_user = matcher.match(node_person_type, name=user).first()
        sims = data.iloc[i][7].spilt('/')
        for sim in sims:
            idx = sim.spilt()[0]
            val = sim.spilt()[1]
            if val > threshhold:
                count += 1
                user_compared = data.loc[idx, 'id']
                node_user_compared = matcher.match(node_person_type, name=user_compared).first()
                properties = {'text similarity': float(val)}
                r = Relationship(node_user, 'textSimilarTo', node_user_compared, **properties)
                relation_list.append(r)
            else:
                break
        if len(relation_list) == 0:
            continue
        relation = Subgraph(relationships=relation_list)
        graph.create(relation)
        relation_list = []
    print('Sentence similarity relation done!!')


def delete_user(dataset='youtube'):
    """
        删除图谱中的用户及其对应关系
        :return:
    """
    if dataset == 'youtube':
        node_person_type = 'YoutubeUser'
    elif dataset == 'pan':
        node_person_type = 'PanUser'
    elif dataset == 'my':
        node_person_type = 'MyUser'

    graph = Graph('http://10.112.170.242:7477', username='neo4j', password='962464')
    graph.run('match (m:{}) - [r] -> () delete r'.format(node_person_type))
    graph.run('match (m:{}) delete m'.format(node_person_type))


def delete_user_personality(dataset='youtube'):
    """
        删除图谱中的用户与人格节点之间的关系
        :return:
    """
    if dataset == 'youtube':
        node_person_type = 'YoutubeUser'
    elif dataset == 'pan':
        node_person_type = 'PanUser'
    elif dataset == 'my':
        node_person_type = 'MyUser'

    graph = Graph('http://10.112.170.242:7477', username='neo4j', password='962464')
    graph.run('match (m:{}) - [r] -> (p:{}) delete r'.format(node_person_type, 'personality'))


def delete_user_similar(dataset='youtube'):
    """
        删除图谱中的用户与人格节点之间的关系
        :return:
    """
    if dataset == 'youtube':
        node_person_type = 'YoutubeUser'
    elif dataset == 'pan':
        node_person_type = 'PanUser'
    elif dataset == 'my':
        node_person_type = 'MyUser'

    graph = Graph('http://dddd', username='neo4j', password='ddd')
    graph.run('match (m:{}) - [r] -> (p:{}) delete r'.format(node_person_type, node_person_type))


def change_kg_from_1_to_2():
    """
    将网络从版本1转换为版本2，即增加ope到ope-low, ope-avg, ope-high（以及另外四种人格）之间的divideInto关系
    :return:
    """
    graph = Graph('http://10.112.49.188:7477', username='neo4j', password='962464')
    matcher = NodeMatcher(graph)
    l = ['ope', 'con', 'agr', 'ext', 'neu']
    for p in l:
        node = matcher.match('personality', name=p).first()
        node1 = matcher.match('personality', name=p + '-low').first()
        node2 = matcher.match('personality', name=p + '-avg').first()
        node3 = matcher.match('personality', name=p + '-high').first()

        r1 = Relationship(node, 'divideInto', node1)
        r2 = Relationship(node, 'divideInto', node2)
        r3 = Relationship(node, 'divideInto', node3)
        graph.create(r1)
        graph.create(r2)
        graph.create(r3)


def change_kg_from_2_to_1():
    """
    将网络从版本2转换为版本1，即删除ope到ope-low, ope-avg, ope-high（以及另外四种人格）之间的divideInto关系
    :return:
    """
    graph = Graph('http://10.112.49.188:7477', username='neo4j', password='962464')
    graph.run('match (m:personality) - [r] -> (n:personality) delete r')


def process_kg_json(json_file, output_path=None):
    """
    处理知识图谱的JSON文件，并从中读取生成嵌入所需要的5个txt文件
    使用前需要手动导出图谱的json格式文件
    :param json_file:图谱的json格式文件路径
    :param output_path: 图谱嵌入所需的文件储存位置
    :return: train2id.txt, valid2id.txt, test2id.txt, relation2id.txt, entity2id.txt
    """
    entity2id = {}
    id2entity = {}
    relation2id = {}
    id2relation = {}
    triple_id = []
    triple_text = []
    entity_label = set()
    relation = set()
    entity = set()

    json_data = []
    with open(json_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            json_data.append(json.loads(line))

    for line in json_data:
        if line['type'] == 'node':
            entity.add(line['properties']['name'])
            entity2id[line['properties']['name']] = int(line['id'])
            id2entity[int(line['id'])] = line['properties']['name']
        elif line['type'] == 'relationship':
            relation.add(line['label'])
            relation2id[line['label']] = int(line['id'])
            id2relation[int(line['id'])] = line['label']

    for line in json_data:
        if line['type'] == 'relationship':
            if line['label'] == 'textSimilarTo':
                triple_text.append((id2entity[int(line['start']['id'])],
                                    id2entity[int(line['end']['id'])],
                                    line['label'],
                                    float(line['properties']['text similarity'])))
            else:
                triple_text.append((id2entity[int(line['start']['id'])],
                                    id2entity[int(line['end']['id'])],
                                    line['label']))

    for num, r in enumerate(relation):
        id2relation[num] = r
        relation2id[r] = num

    for num, e in enumerate(entity):
        id2entity[num] = e
        entity2id[e] = num

    for triple in triple_text:
        if len(triple) == 4:
            start, end, rel, sim = triple
            triple_id.append((entity2id[start], entity2id[end], relation2id[rel], sim))
        else:
            start, end, rel = triple
            triple_id.append((entity2id[start], entity2id[end], relation2id[rel]))

    random.shuffle(triple_id)
    length = len(triple_id)
    train = triple_id[:int(0.8 * length)]
    valid = triple_id[int(0.8 * length): int(0.9 * length)]
    test = triple_id[int(0.9 * length):]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, 'triple2id.txt'), 'w') as f:
        f.write('{}\n'.format(len(triple_id)))
        for triple in triple_id:
            if len(triple) == 4:
                f.write('{}   {}   {}   {}\n'.format(triple[0], triple[1], triple[2], triple[3]))
            else:
                f.write('{}   {}   {}\n'.format(triple[0], triple[1], triple[2]))

    with open(os.path.join(output_path, 'train2id.txt'), 'w') as f:
        f.write('{}\n'.format(len(train)))
        for triple in train:
            f.write('{}   {}   {}\n'.format(triple[0], triple[1], triple[2]))

    with open(os.path.join(output_path, 'valid2id.txt'), 'w') as f:
        f.write('{}\n'.format(len(valid)))
        for triple in valid:
            f.write('{}   {}   {}\n'.format(triple[0], triple[1], triple[2]))

    with open(os.path.join(output_path, 'test2id.txt'), 'w') as f:
        f.write('{}\n'.format(len(test)))
        for triple in test:
            f.write('{}   {}   {}\n'.format(triple[0], triple[1], triple[2]))

    with open(os.path.join(output_path, 'entity2id.txt'), 'w') as f:
        f.write('{}\n'.format(len(entity2id)))
        for k, v in entity2id.items():
            f.write('{}   {}\n'.format(k, v))

    with open(os.path.join(output_path, 'relation2id.txt'), 'w') as f:
        f.write('{}\n'.format(len(relation2id)))
        for k, v in relation2id.items():
            f.write('{}   {}\n'.format(k, v))


def train(inpath, dataset, embedding_size, train_rate, split_rate, version):
    """
    根据指定的数据集，使用从图谱中生成的5个txt文件进行训练，并保存ckpt文件
    如果改变训练函数中的超参数，则在获取向量函数中也要修改，保持一致
    :param dataset:
    :param embedding_size: 向量维度
    :param train_rate:训练集比例
    :param split_rate: 分割比例
    :param version: 图谱版本
    :return:
    """

    train_dataloader = TrainDataLoader(
        in_path=inpath,
        nbatches=100,
        threads=8,
        sampling_mode='normal',
        bern_flag=1,
        filter_flag=1,
        neg_ent=1,
        neg_rel=1)

    test_dataloader = TestDataLoader(inpath, 'link')

    transd = TransD(ent_tot=train_dataloader.get_ent_tot(),
                    rel_tot=train_dataloader.get_rel_tot(),
                    dim_e=embedding_size,
                    dim_r=embedding_size,
                    norm_flag=True)

    model = NegativeSampling(model=transd,
                             loss=MarginLoss(margin=4.0),
                             batch_size=train_dataloader.get_batch_size())

    # train the model
    trainer = Trainer(model=model, data_loader=train_dataloader, train_times=1000, alpha=0.05, use_gpu=True)
    trainer.run()
    checkpoint_path = './checkpoint/transd_' + '{}_{}_{}_{}_'.format(dataset, train_rate, split_rate, version) \
                      + str(embedding_size) + '.ckpt'
    transd.save_checkpoint(checkpoint_path)

    # test the model
    transd.load_checkpoint(checkpoint_path)
    tester = Tester(model=transd, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)


def get_entity_embedding(inpath, dataset, embedding_size, train_rate, split_rate, version):
    """
    获取指定数据集训练模型中的嵌入向量矩阵
    如果改变train函数中的超参数，则在此函数中也要修改，保持一致
    :param dataset:
    :param embedding_size:
    :param rate: 训练集比例
    :return: numpy类型的向量矩阵
    """

    train_dataloader = TrainDataLoader(
        in_path=inpath,
        nbatches=100,
        threads=8,
        sampling_mode='normal',
        bern_flag=1,
        filter_flag=1,
        neg_ent=1,
        neg_rel=0)

    transd = TransD(ent_tot=train_dataloader.get_ent_tot(),
                    rel_tot=train_dataloader.get_rel_tot(),
                    dim_e=embedding_size,
                    dim_r=embedding_size,
                    norm_flag=True)

    checkpoint_path = './checkpoint/transd_' + '{}_{}_{}_{}_'.format(dataset, train_rate, split_rate, version) \
                      + str(embedding_size) + '.ckpt'

    transd.load_checkpoint(checkpoint_path)
    ent_embedding = transd.ent_embeddings.weight.data.numpy()
    rel_embedding = transd.rel_embeddings.weight.data.numpy()
    ent_transfer = transd.ent_transfer.weight.data.numpy()
    rel_transfer = transd.rel_transfer.weight.data.numpy()
    return ent_embedding, rel_embedding, ent_transfer, rel_transfer


def get_hashmap(inpath):
    """
    获取指定数据集的entity2id字典， relation2id字典
    :param dataset:
    :return:
    """
    entity2id = {}
    relation2id = {}
    file_entity = inpath + 'entity2id.txt'
    file_relation = inpath + 'relation2id.txt'

    with open(file_entity) as f:
        print('total num of entity: ', f.readline())
        for line in f.readlines():
            line = line.split()
            name, seq = line[0], line[1]
            entity2id[str(name)] = int(seq)

    with open(file_relation) as f:
        print('total num of relation: ', f.readline())
        for line in f.readlines():
            line = line.split()
            name, seq = line[0], line[1]
            relation2id[str(name)] = int(seq)
    return entity2id, relation2id


def get_trainset(file_path, inpath, dataset, embedding_size, train_rate, split_rate, version):
    """
    根据指定的数据集和对应的维度，生成训练集
    :param file_path:'/home/gzm/personality/dataset/youtube_liwc.csv'
    :param inpath: 储存entity2id的文件夹
    :param dataset:
    :param embedding_size:
    :param rate:
    :return: [n * (embedding_size + 5)]
    """

    train_list_file = 'dataset/trainlist_{}_{}_{}_{}.json'.format(dataset, train_rate, split_rate, version)

    trainList = []
    with open(train_list_file, 'r') as f:
        trainList = json.load(f)

    data = pd.read_csv(file_path)
    allList = list(data.id.values)
    predictList = [item for item in allList if item not in trainList]
    data = data[data['id'].isin(predictList)]
    data = data.loc[:, ['id', 'ope', 'con', 'ext', 'agr', 'neu']]
    entity2id, relation2id = get_hashmap(inpath)
    ent_embedding, rel_embedding, ent_transfer, rel_transfer = get_entity_embedding(inpath, dataset,
                                                                                    embedding_size, train_rate,
                                                                                    split_rate, version)

    trainset = []
    for i in range(data.shape[0]):
        user = list(data.iloc[i, :])
        user_name, user_label = user[0], user[1:]
        line = entity2id.get(user_name, None)
        if line is None:
            continue
        else:
            user_embed = list(ent_embedding[line, :])
            trainset.append(user_embed + user_label)
    return np.array(trainset)


def get_trainset_specified_label(inpath, dataset, embedding_size, label, train_rate, split_rate, version):
    """
    根据指定的数据集和对应的维度，生成训练集
    :param dataset:
    :param embedding_size:
    :param rate:
    :param label:针对某一种人格生成训练集
    :return: [n * (embedding_size + 5)]
    """
    if dataset == 'youtube':
        datapath = '../datasets/youtube_for_kg.csv'
    elif dataset == 'pan':
        datapath = '../datasets/pan_for_kg.csv'
    elif dataset == 'my':
        datapath = '../datasets/my_for_kg.csv'

    train_list_file = 'dataset/trainlist_{}_{}_{}_{}.json'.format(dataset, train_rate, split_rate, version)

    with open(train_list_file, 'r') as f:
        trainList = json.load(f)
        print('trainList length: ', len(trainList))

    data = pd.read_csv(datapath)
    if dataset == 'my':
        data = data[:10000]
    allList = list(data.id.values)
    predictList = [item for item in allList if item not in trainList]

    train_data = data[data['id'].isin(trainList)]
    train_data = train_data.loc[:, ['id', label]]
    test_data = data[data['id'].isin(predictList)]
    test_data = test_data.loc[:, ['id', label]]

    entity2id, relation2id = get_hashmap(inpath)
    ent_embedding, rel_embedding, ent_transfer, rel_transfer = get_entity_embedding(inpath, dataset,
                                                                                    embedding_size, train_rate,
                                                                                    split_rate, version)
    if label == 'ope':
        relationId = relation2id['opeIs']
    elif label == 'con':
        relationId = relation2id['conIs']
    elif label == 'ext':
        relationId = relation2id['extIs']
    elif label == 'agr':
        relationId = relation2id['agrIs']
    elif label == 'neu':
        relationId = relation2id['neuIs']

    train_user_index_list = [entity2id[u] for u in trainList]
    test_user_index_list = [entity2id[u] for u in predictList]

    train_user_embed = ent_embedding[train_user_index_list]
    train_user_transfer = ent_transfer[train_user_index_list]
    train_rel_embed = rel_embedding[relationId]
    train_rel_transfer = rel_transfer[relationId]
    # train_labels = train_data.loc[:, [label]]
    train_labels = []
    for id in tqdm(trainList):
        train_labels.append(train_data[(train_data.id == id)].loc[:, label].values[0])
    train_labels = np.asarray(train_labels)
    train_labels = np.expand_dims(train_labels, 1)

    train = transfer(train_user_embed, train_user_transfer, train_rel_embed, train_rel_transfer)
    train = np.concatenate((train, train_labels), axis=1)

    test_user_embed = ent_embedding[test_user_index_list]
    test_user_transfer = ent_transfer[test_user_index_list]
    test_rel_embed = rel_embedding[relationId]
    test_rel_transfer = rel_transfer[relationId]
    # test_labels = test_data.loc[:, [label]]
    test_labels = []
    for id in tqdm(predictList):
        test_labels.append(test_data[(test_data.id == id)].loc[:, label].values[0])
    test_labels = np.asarray(test_labels)
    test_labels = np.expand_dims(test_labels, 1)
    test = transfer(test_user_embed, test_user_transfer, test_rel_embed, test_rel_transfer)
    test = np.concatenate((test, test_labels), axis=1)

    return train, test


def transfer(h, hp, r, rp):
    """
    将四个向量进行转换，生成最终进行训练的训练集
    :param h: entity embedding [batch_size, embedding_size]
    :param hp: entity projection embedding [batch_size, embedding_size]
    :param r: relation embedding [embedding_size, ]
    :param rp: relation projection embedding [embedding_size, ]
    :return: train vectors [batch_size, embedding_size]
    """
    r = np.expand_dims(r, 0).repeat(h.shape[0], axis=0)
    rp = np.expand_dims(rp, 0).repeat(h.shape[0], axis=0)
    h_projected = np.sum(h * hp, axis=-1, keepdims=True) * rp + h
    # h_projected = normalize(h_projected, norm='l2', axis=1)
    # return h_projected + r
    return h_projected
    # return h


def get_semantic(dataset, train_rate, split_rate, version):
    if dataset == 'youtube':
        datapath = '../datasets/youtube_for_kg.csv'
    elif dataset == 'pan':
        datapath = '../datasets/pan_for_kg.csv'
    elif dataset == 'my':
        datapath = '../datasets/my_for_kg.csv'

    train_list_file = 'dataset/trainlist_{}_{}_{}_{}.json'.format(dataset, train_rate, split_rate, version)

    print('loading bert tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    with open(train_list_file, 'r') as f:
        trainList = json.load(f)
        print('trainList length: ', len(trainList))

    data = pd.read_csv(datapath)
    if dataset == 'my':
        data = data[:10000]

    allList = list(data.id.values)
    predictList = [item for item in allList if item not in trainList]
    train_data = data[data['id'].isin(trainList)]
    train_data = train_data.loc[:, ['id', 'posts']]
    test_data = data[data['id'].isin(predictList)]
    test_data = test_data.loc[:, ['id', 'posts']]
    test_data = test_data.reset_index(drop=True)

    train_semantic_features = []
    test_semantic_features = []
    print('generating train semantic features...')
    for id in tqdm(trainList):
        posts = train_data[(train_data.id == id)].posts.values[0]
        train_semantic_features.append(get_bert_embeddings(posts, model, tokenizer))

    print('generating test semantic features...')
    for id in tqdm(predictList):
        posts = test_data[(test_data.id == id)].posts.values[0]
        test_semantic_features.append(get_bert_embeddings(posts, model, tokenizer))

    train_semantic_features = t.cat(train_semantic_features, dim=0).numpy()
    test_semantic_features = t.cat(test_semantic_features, dim=0).numpy()

    return train_semantic_features, test_semantic_features


def get_bert_embeddings(posts, model, tokenizer):
    posts = [post.strip() for post in posts.split('<sep>') if len(post.strip()) != 0]

    input_ids = []
    attention_masks = []
    for post in posts:
        encoded_dict = tokenizer.encode_plus(post, add_special_tokens=True, max_length=200,
                                             pad_to_max_length=True, return_attention_mask=True,
                                             return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = t.cat(input_ids, dim=0)
    with t.no_grad():
        hidden, pooler = model(input_ids)
    return t.mean(pooler, dim=0).unsqueeze(0)


'''
new version
'''


def single_score_SVR(metrics, dataset, file_name, embedding_size, train_rate, split_rate, version):
    inpath = 'kg_embedding_dataset/{}/'.format(file_name)
    label_list = ['ope', 'con', 'ext', 'agr', 'neu']
    result_kg = {}
    # result_semantic = {}
    # result_both = {}
    # train_semantic, test_semantic = get_semantic(dataset, train_rate, split_rate, version)
    for label in label_list:
        print('正在训练{}人格回归模型...'.format(label))

        train_kg, test_kg = get_trainset_specified_label(inpath, dataset, embedding_size, label,
                                                         train_rate, split_rate, version)
        input_train_kg = train_kg[:, :-1]
        label_train = np.squeeze(train_kg[:, -1:], axis=1)
        input_test_kg = test_kg[:, :-1]
        label_test = np.squeeze(test_kg[:, -1:], axis=1)

        # 使用语义向量和图谱向量拼接做SVR预测
        # train_both = np.concatenate((train_semantic, input_train_kg), axis=1)
        # test_both = np.concatenate((test_semantic, input_test_kg), axis=1)
        # clf_both = svm.SVR(gamma='scale')
        # # clf = svm.SVR(gamma='scale')
        # clf_both.fit(train_both, label_train)
        # y_pred_both = clf_both.predict(test_both)

        # 使用图谱向量做SVR预测
        clf_kg = svm.SVR(gamma='scale')
        # clf = svm.SVR(gamma='scale')
        clf_kg.fit(input_train_kg, label_train)
        y_pred_kg = clf_kg.predict(input_test_kg)
        # random guess
        # y_pred =  0.6 * np.random.random((147, 5)) - 0.1

        # 使用语义向量做SVR预测
        # clf_semantic = svm.SVR(gamma='scale')
        # # clf = svm.SVR(gamma='scale')
        # clf_semantic.fit(train_semantic, label_train)
        # y_pred_semantic = clf_semantic.predict(test_semantic)

        if metrics == 'MSE':
            result_kg[label] = mean_squared_error(label_test, y_pred_kg)
            # result_semantic[label] = mean_squared_error(label_test, y_pred_semantic)
            # result_both[label] = mean_squared_error(label_test, y_pred_both)
        elif metrics == 'MAE':
            result_kg[label] = mean_absolute_error(label_test, y_pred_kg)
            # result_semantic[label] = mean_absolute_error(label_test, y_pred_semantic)
            # result_both[label] = mean_absolute_error(label_test, y_pred_both)

    return result_kg


def regression_SVR(train_x, train_label, test_x, test_label, metric='MSE'):
    clf = MultiOutputRegressor(svm.SVR(gamma='scale'))
    # clf = svm.SVR(gamma='scale')
    clf.fit(train_x, train_label)
    y_pred = clf.predict(test_x)
    # random guess
    # y_pred =  0.6 * np.random.random((147, 5)) - 0.1
    if metric == 'MSE':
        M = mean_squared_error(test_label, y_pred, multioutput='raw_values')
    elif metric == 'MAE':
        M = mean_absolute_error(test_label, y_pred, multioutput='raw_values')
    else:
        M = np.sqrt(mean_squared_error(test_label, y_pred, multioutput='raw_values'))

    # print(type(MSE))
    result = []
    for i in range(5):
        result.append(M[i])
    return result


def score_SVR(metrics, dataset, file_name, embedding_size, test_size, train_rate, split_rate, version):
    score_times = 10
    inpath = 'kg_embedding_dataset/{}/'.format(file_name)
    if dataset == 'youtube':
        file_path = '../datasets/youtube_for_kg.csv'
    elif dataset == 'pan':
        file_path = '../datasets/pan_for_kg.csv'
    elif dataset == 'my':
        file_path = '../datasets/my_for_kg.csv'
    # all_results = []
    # min_score = []
    # max_score = []
    mean_score = []

    data = get_trainset(file_path,inpath, dataset, embedding_size, train_rate, split_rate, version)
    data = data.astype(np.float64)

    total = score_times
    count = 0

    results = []
    for times in range(score_times):
        count = count + 1
        print('finished %.4f / 100' % (count * 1.0 / total * 10))
        train, test = train_test_split(data, test_size=test_size, shuffle=True)
        # train = data[:250, :]
        # test = data[250:, :]
        x_train = train[:, :-5]
        y_train = train[:, -5:]
        x_test = test[:, :-5]
        y_test = test[:, -5:]
        result = regression_SVR(x_train, y_train, x_test, y_test, metrics)
        results.append(result)
    print(results)
    results = pd.DataFrame(results)
    # max_score.append(results.max())
    mean_score.append(results.mean())
    # all_results.append(results)

    # all_results = pd.DataFrame(all_results)
    # min_score = pd.DataFrame(min_score)
    # max_score = pd.DataFrame(max_score)
    mean_score = pd.DataFrame(mean_score)

    return mean_score


def regression_XGBOOST(train_x, train_label, test_x, test_label, metric='MSE'):
    clf = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:gamma',
                                                booster='gbtree',
                                                gamma=0.1,
                                                max_depth=5,
                                                reg_lambda=3,
                                                subsample=0.7,
                                                min_child_weight=3))
    # clf = svm.SVR(gamma='scale')
    clf.fit(train_x, train_label)
    y_pred = clf.predict(test_x)
    # random guess
    # y_pred =  0.6 * np.random.random((147, 5)) - 0.1
    if metric == 'MSE':
        M = mean_squared_error(test_label, y_pred, multioutput='raw_values')
    elif metric == 'MAE':
        M = mean_absolute_error(test_label, y_pred, multioutput='raw_values')
    else:
        M = np.sqrt(mean_squared_error(test_label, y_pred, multioutput='raw_values'))

    # print(type(MSE))
    result = []
    for i in range(5):
        result.append(M[i])
    return result


def score_XGBOOST(metrics, dataset, embedding_size):
    score_times = 10

    all_results = []
    min_score = []
    max_score = []
    mean_score = []

    data = get_trainset(dataset, embedding_size)
    data = data.astype(np.float64)

    total = score_times
    count = 0

    results = []
    for times in range(score_times):
        count = count + 1
        print('finished %.4f / 100' % (count * 1.0 / total * 10))
        train, test = train_test_split(data, test_size=0.5, shuffle=True)
        # train = data[:250, :]
        # test = data[250:, :]
        x_train = train[:, :-5]
        y_train = train[:, -5:]
        x_test = test[:, :-5]
        y_test = test[:, -5:]
        result = regression_XGBOOST(x_train, y_train, x_test, y_test, metrics)
        results.append(result)
    print(results)
    results = pd.DataFrame(results)
    min_score.append(results.min())
    max_score.append(results.max())
    mean_score.append(results.mean())
    all_results.append(results)

    all_results = pd.DataFrame(all_results)
    min_score = pd.DataFrame(min_score)
    max_score = pd.DataFrame(max_score)
    mean_score = pd.DataFrame(mean_score)

    return all_results, max_score, min_score, mean_score


def ml_test():
    """
    使用机器学习方法进行测试
    :return:
    """
    all, max, min, mean = score_XGBOOST('MSE')
    print('max:', max)
    print('min:', min)
    print('mean:', mean)


def single_regression_SVR(train_x, train_label, test_x, test_label, metric='MSE'):
    """

    :param train_x: [m * embed_size]
    :param train_label: [m * 1]
    :param test_x: [n * embed_size]
    :param test_label: [n * 1]
    :param metric:
    :return:
    """
    clf = svm.SVR(gamma='scale')
    print(clf.get_params())
    # clf = svm.SVR(gamma='scale')
    clf.fit(train_x, train_label)
    y_pred = clf.predict(test_x)
    # random guess
    # y_pred =  0.6 * np.random.random((147, 5)) - 0.1
    if metric == 'MSE':
        M = mean_squared_error(test_label, y_pred)
    elif metric == 'MAE':
        M = mean_absolute_error(test_label, y_pred)
    else:
        M = np.sqrt(mean_squared_error(test_label, y_pred))

    # print(type(MSE))
    return M


'''
使用网格搜索确定最优参数组合
'''


# TODO 需要重写改方法
def GridSearch_SVR(metrics, dataset, embedding_size, rate):
    label_list = ['ope', 'con', 'ext', 'agr', 'neu']
    result = {}
    for label in label_list:
        print('正在训练{}人格回归模型...'.format(label))
        train, test = get_trainset_specified_label(dataset, embedding_size, rate, label)
        x_train = train[:, :-1]
        y_train = np.squeeze(train[:, -1:], axis=1)
        x_test = test[:, :-1]
        y_test = np.squeeze(test[:, -1:], axis=1)
        # result[label] = single_regression_SVR(x_train, y_train, x_test, y_test, metrics)

        svr = svm.SVR(gamma='scale')
        params = {}
        gc = GridSearchCV(svr, param_grid=params, scoring='neg_mean_absolute_error')
        gc.fit(x_train, y_train)
        print('grid search for {}:\n'.format(label))
        gc.score(x_test, y_test)
        print('best score: {}\n'.format(gc.best_score_))
        print('best model: {}\n'.format(gc.best_estimator_))
        # print('all results: {}\n'.format(gc.cv_results_))
        result[label] = gc.best_score_
    return result


'''
MLP训练
'''


def single_score_MLP(metrics, dataset, embedding_size, rate):
    label_list = ['ope', 'con', 'ext', 'agr', 'neu']
    result = {}
    for label in label_list:
        print('正在训练{}人格回归模型...'.format(label))
        train, test = get_trainset_specified_label(dataset, embedding_size, rate, label)
        x_train = train[:, :-1]
        y_train = np.squeeze(train[:, -1:], axis=1)
        x_test = test[:, :-1]
        y_test = np.squeeze(test[:, -1:], axis=1)
        # result[label] = single_regression_SVR(x_train, y_train, x_test, y_test, metrics)

        mlp = MLPRegressor(hidden_layer_sizes=(200, 100, 100), activation='relu', solver='adam',
                           alpha=0.001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                           power_t=0.5, max_iter=200, tol=1e-4, early_stopping=True)
        mlp.fit(x_train, y_train)
        y_pred = mlp.predict(x_test)
        if metrics == 'MSE':
            M = mean_squared_error(y_test, y_pred)
        elif metrics == 'MAE':
            M = mean_absolute_error(y_test, y_pred)
        else:
            M = np.sqrt(mean_squared_error(y_test, y_pred))
        result[label] = M
    return result
