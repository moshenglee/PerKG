from py2neo import *
from transformers import AlbertTokenizer,AlbertModel,BertTokenizer, BertModel
from py2neo import Node, Relationship
from py2neo import Graph, NodeMatcher, Subgraph
import pandas as pd
import pickle
import json
import os
import random
from openke.config import Trainer, Tester
from openke.module.model import TransD
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import math
import time
import xgboost as xgb
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch as t
import sys
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from torch.nn.functional import avg_pool2d
from gensim.models import word2vec


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
    graph = Graph('http://10.112.49.188:7477', username='neo4j', password='111111')
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
    graph = Graph('http://10.112.49.188:7477', username='neo4j', password='111111')
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
    graph = Graph('http://10.112.49.188:7477', username='neo4j', password='111111')
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

    graph = Graph('http://10.112.49.188:7477', username='neo4j', password='111111')
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

    graph = Graph('http://10.112.49.188:7477', username='neo4j', password='111111')
    graph.run('match (m:{}) - [r] -> (p:{}) delete r'.format(node_person_type, 'personality'))


def change_kg_from_1_to_2():
    """
    将网络从版本1转换为版本2，即增加ope到ope-low, ope-avg, ope-high（以及另外四种人格）之间的divideInto关系
    :return:
    """
    graph = Graph('http://10.112.49.188:7477', username='neo4j', password='111111')
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
    graph = Graph('http://10.112.49.188:7477', username='neo4j', password='111111')
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
        start, end, rel = triple
        triple_id.append((entity2id[start], entity2id[end], relation2id[rel]))

    random.shuffle(triple_id)
    length = len(triple_id)
    train = triple_id[:int(0.8 * length)]
    valid = triple_id[int(0.8 * length): int(0.9 * length)]
    test = triple_id[int(0.9 * length):]

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

    train_list_file = 'dataset/trainlist_{}_{}_{}.json'.format(dataset, train_rate, split_rate, version)

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


def get_semantic(dataset, label, file_name, train_rate, split_rate, version):
    if dataset == 'youtube':
        datapath = '../datasets/youtube_for_kg.csv'
    elif dataset == 'pan':
        datapath = '../datasets/pan_for_kg.csv'
    elif dataset == 'my':
        datapath = '../datasets/my_for_kg.csv'

    train_list_file = 'dataset/trainlist_{}_{}_{}_{}.json'.format(dataset, train_rate, split_rate, version)
    model_dir = 'SEMANTIC/model_save/{}/{}'.format(file_name, label)

    print('训练名单文件:{}'.format(train_list_file))
    print('模型所在路径:{}'.format(model_dir))

    print('loading bert tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)
    model = BertModel.from_pretrained(model_dir)
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
    posts = [post.strip() for post in posts.split('<SEP>') if len(post.strip()) != 0]

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


def single_score_SVR(metrics, dataset, file_name, embedding_size, train_rate, split_rate, version, device_id,
                     batch_size):
    inpath = 'kg_embedding_dataset/{}/'.format(file_name)
    label_list = ['ope', 'con', 'ext', 'agr', 'neu']
    result_kg_mae = {}
    result_semantic_mae = {}
    result_both_mae = {}

    result_kg_mse = {}
    result_semantic_mse = {}
    result_both_mse = {}
    for label in label_list:
        print('正在训练{}人格回归模型...'.format(label))

        # train_kg, test_kg = get_trainset_specified_label(inpath, dataset, embedding_size, label,
        #                                            train_rate, split_rate, version)

        train, test = get_features_specified_label(inpath, dataset, file_name, embedding_size, label,
                                                   train_rate, split_rate, version, device_id, batch_size)

        train_semantic, train_kg, train_label = train
        test_semantic, test_kg, test_label = test

        # 使用语义向量和图谱向量拼接做SVR预测
        train_both = np.concatenate((train_semantic, train_kg), axis=1)
        test_both = np.concatenate((test_semantic, test_kg), axis=1)
        clf_both = svm.SVR(gamma='scale')
        # clf = svm.SVR(gamma='scale')
        clf_both.fit(train_both, train_label)
        y_pred_both = clf_both.predict(test_both)

        # 使用图谱向量做SVR预测
        clf_kg = svm.SVR(gamma='scale')
        # clf = svm.SVR(gamma='scale')
        clf_kg.fit(train_kg, train_label)
        y_pred_kg = clf_kg.predict(test_kg)
        # random guess
        # y_pred =  0.6 * np.random.random((147, 5)) - 0.1

        # 使用语义向量做SVR预测
        clf_semantic = svm.SVR(gamma='scale')
        # clf = svm.SVR(gamma='scale')
        clf_semantic.fit(train_semantic, train_label)
        y_pred_semantic = clf_semantic.predict(test_semantic)

        result_kg_mse[label] = mean_squared_error(test_label, y_pred_kg)
        result_semantic_mse[label] = mean_squared_error(test_label, y_pred_semantic)
        result_both_mse[label] = mean_squared_error(test_label, y_pred_both)

        result_kg_mae[label] = mean_absolute_error(test_label, y_pred_kg)
        result_semantic_mae[label] = mean_absolute_error(test_label, y_pred_semantic)
        result_both_mae[label] = mean_absolute_error(test_label, y_pred_both)

    return result_kg_mse, result_semantic_mse, result_both_mse, result_kg_mae, result_semantic_mae, result_both_mae


'''
MLP
'''


def single_score_MLP(metrics, dataset, file_name, embedding_size, train_rate, split_rate, version, device_id,
                     batch_size):
    inpath = 'kg_embedding_dataset/{}/'.format(file_name)
    label_list = ['ope', 'con', 'ext', 'agr', 'neu']
    result_kg_mae = {}
    result_semantic_mae = {}
    result_both_mae = {}

    result_kg_mse = {}
    result_semantic_mse = {}
    result_both_mse = {}
    for label in label_list:
        print('正在训练{}人格回归模型...'.format(label))

        # train_kg, test_kg = get_trainset_specified_label(inpath, dataset, embedding_size, label,
        #                                            train_rate, split_rate, version)

        train, test = get_features_specified_label(inpath, dataset, file_name, embedding_size, label,
                                                   train_rate, split_rate, version, device_id, batch_size)

        train_semantic, train_kg, train_label = train
        test_semantic, test_kg, test_label = test

        # 使用语义向量和图谱向量拼接做SVR预测
        train_both = np.concatenate((train_semantic, train_kg), axis=1)
        test_both = np.concatenate((test_semantic, test_kg), axis=1)
        # clf_both = svm.SVR(gamma='scale')
        mlp_both = MLPRegressor((100, 50))
        # clf = svm.SVR(gamma='scale')
        mlp_both.fit(train_both, train_label)
        y_pred_both = mlp_both.predict(test_both)

        # 使用图谱向量做SVR预测
        # clf_kg = svm.SVR(gamma='scale')
        mlp_kg = MLPRegressor()
        # clf = svm.SVR(gamma='scale')
        mlp_kg.fit(train_kg, train_label)
        y_pred_kg = mlp_kg.predict(test_kg)
        # random guess
        # y_pred =  0.6 * np.random.random((147, 5)) - 0.1

        # 使用语义向量做SVR预测
        # clf_semantic = svm.SVR(gamma='scale')
        mlp_semantic = MLPRegressor()
        # clf = svm.SVR(gamma='scale')
        mlp_semantic.fit(train_semantic, train_label)
        y_pred_semantic = mlp_semantic.predict(test_semantic)

        result_kg_mse[label] = mean_squared_error(test_label, y_pred_kg)
        result_semantic_mse[label] = mean_squared_error(test_label, y_pred_semantic)
        result_both_mse[label] = mean_squared_error(test_label, y_pred_both)

        result_kg_mae[label] = mean_absolute_error(test_label, y_pred_kg)
        result_semantic_mae[label] = mean_absolute_error(test_label, y_pred_semantic)
        result_both_mae[label] = mean_absolute_error(test_label, y_pred_both)

    return result_kg_mse, result_semantic_mse, result_both_mse, result_kg_mae, result_semantic_mae, result_both_mae


def get_features_specified_label(inpath, dataset, file_name, embedding_size, label, train_rate, split_rate, version,
                                 device_id, batch_size):
    """

    :param hashmappath:
    :param dataset:
    :param file_name:
    :param embedding_size:
    :param label:
    :param train_rate:
    :param split_rate:
    :param version:
    :return:
    """

    '''
    确定数据集文件
    '''
    if dataset == 'youtube':
        datapath = '../datasets/youtube_for_kg.csv'
    elif dataset == 'pan':
        datapath = '../datasets/pan_for_kg.csv'
    elif dataset == 'my':
        datapath = '../datasets/my_for_kg.csv'

    '''
    训练集名单和预训练的模型地址
    '''
    train_list_file = 'dataset/trainlist_{}_{}_{}_{}.json'.format(dataset, train_rate, split_rate, version)
    # TODO change to albert
    model_dir = 'SEMANTIC/model_save/{}/{}'.format(file_name, label)

    print('训练名单文件:{}'.format(train_list_file))
    print('模型所在路径:{}'.format(model_dir))

    print('loading bert tokenizer...')
    tokenizer = AlbertTokenizer.from_pretrained(model_dir, do_lower_case=True)
    model = AlbertModel.from_pretrained(model_dir)
    model.eval()

    if t.cuda.is_available():
        device = t.device('cuda')
        print('there are %d GPU available.' % t.cuda.device_count())
        print('we will use the gpu: ', t.cuda.get_device_name())
    else:
        print('no gpu available, using cpu instead.')
        device = t.device('cpu')
    t.cuda.set_device(device_id)

    model.cuda(device)
    print(next(model.parameters()).is_cuda)

    with open(train_list_file, 'r') as f:
        trainList = json.load(f)
        print('trainList length: ', len(trainList))

    data = pd.read_csv(datapath)
    if dataset == 'my':
        data = data[:12000]
    allList = list(data.id.values)
    predictList = [item for item in allList if item not in trainList]

    print(len(allList))
    print(len(predictList))

    train_data = data[data['id'].isin(trainList)]
    train_data = train_data.loc[:, ['id', 'posts', label]]
    test_data = data[data['id'].isin(predictList)]
    test_data = test_data.loc[:, ['id', 'posts', label]]

    split_flag = '<sep>'
    if dataset == 'my':
        split_flag = '<SEP>'

    print('number of train users: {:,}\n'.format(train_data.shape[0]))
    print('number of test users: {:,}\n'.format(test_data.shape[0]))

    entity2id, relation2id = get_hashmap(inpath)
    print('length of entity2id: {}'.format(len(entity2id)))

    train_input, train_mask, train_id, train_rel_id, train_label = \
        process_data_into_tensor(train_data, label, tokenizer, entity2id, relation2id, split_flag, post_num=32,
                                 post_length=16)
    test_input, test_mask, test_id, test_rel_id, test_label = \
        process_data_into_tensor(test_data, label, tokenizer, entity2id, relation2id, split_flag, post_num=32,
                                 post_length=16)

    train_dataset = TensorDataset(train_input, train_mask, train_id, train_rel_id, train_label)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    test_dataset = TensorDataset(test_input, test_mask, test_id, test_rel_id, test_label)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    print('traindataloader lenght: {}'.format(len(train_dataloader)))

    print('testdataloader lenght: {}'.format(len(test_dataloader)))

    e_embed, r_embed, e_transfer, r_transfer = get_entity_embedding(inpath, dataset, embedding_size, train_rate,
                                                                    split_rate, version)

    entity_embedding = nn.Embedding(e_embed.shape[0], e_embed.shape[1])
    entity_embedding.weight.data.copy_(t.from_numpy(e_embed))
    for param in entity_embedding.parameters():
        param.requires_grad = False
    entity_embedding.cuda(device)

    relation_embedding = nn.Embedding(r_embed.shape[0], r_embed.shape[1])
    relation_embedding.weight.data.copy_(t.from_numpy(r_embed))
    for param in relation_embedding.parameters():
        param.requires_grad = False
    relation_embedding.cuda(device)

    entity_transfer = nn.Embedding(e_transfer.shape[0], e_transfer.shape[1])
    entity_transfer.weight.data.copy_(t.from_numpy(e_transfer))
    for param in entity_transfer.parameters():
        param.requires_grad = False
    entity_transfer.cuda(device)

    relation_transfer = nn.Embedding(r_transfer.shape[0], r_transfer.shape[1])
    relation_transfer.weight.data.copy_(t.from_numpy(r_transfer))
    for param in relation_transfer.parameters():
        param.requires_grad = False
    relation_transfer.cuda(device)

    train_semantic = None
    train_kg = None

    for batch in train_dataloader:
        batch = tuple(i.to(device) for i in batch)
        input, mask, user_id, rel_id, label = batch

        # 生成语义特征
        with t.no_grad():
            b_size, s_len = input.shape[0], input.shape[1]
            input = input.view(b_size * s_len, -1)
            mask = mask.view(b_size * s_len, -1)
            outputs = model(
                input,
                attention_mask=mask,
            )
            # tuple, len=2
            pooled_output = outputs[1]
            # pooled_output [batch_size, 768]
            pooled_output = pooled_output.view(b_size, s_len, -1)
            pooled_output = avg_pool2d(pooled_output, (s_len, 1)).squeeze(1)
        if train_semantic is None:
            train_semantic = pooled_output
        else:
            train_semantic = t.cat((train_semantic, pooled_output), 0)

        # 生成图谱特征
        with t.no_grad():
            e = entity_embedding(user_id)
            r = relation_embedding(rel_id)

            e_t = entity_transfer(user_id)
            r_t = relation_transfer(rel_id)

            kg_output = transfer_torch(e, e_t, r, r_t)
        if train_kg is None:
            train_kg = kg_output
        else:
            train_kg = t.cat((train_kg, kg_output), 0)

    train_semantic = train_semantic.cpu().numpy()
    train_kg = train_kg.cpu().numpy()
    train_label = train_label.numpy()

    test_semantic = None
    test_kg = None

    for batch in test_dataloader:
        batch = tuple(i.to(device) for i in batch)
        input, mask, user_id, rel_id, label = batch

        # 生成语义特征
        with t.no_grad():
            b_size, s_len = input.shape[0], input.shape[1]
            input = input.view(b_size * s_len, -1)
            mask = mask.view(b_size * s_len, -1)
            outputs = model(input, attention_mask=mask)
            # tuple, len=2
            pooled_output = outputs[1]
            # pooled_output [batch_size, 768]
            pooled_output = pooled_output.view(b_size, s_len, -1)
            pooled_output = avg_pool2d(pooled_output, (s_len, 1)).squeeze(1)
        if test_semantic is None:
            test_semantic = pooled_output
        else:
            test_semantic = t.cat((test_semantic, pooled_output), 0)

        # 生成图谱特征
        with t.no_grad():
            e = entity_embedding(user_id)
            r = relation_embedding(rel_id)

            e_t = entity_transfer(user_id)
            r_t = relation_transfer(rel_id)

            kg_output = transfer_torch(e, e_t, r, r_t)
        if test_kg is None:
            test_kg = kg_output
        else:
            test_kg = t.cat((test_kg, kg_output), 0)

    test_semantic = test_semantic.cpu().numpy()
    test_kg = test_kg.cpu().numpy()
    test_label = test_label.numpy()

    return (train_semantic, train_kg, train_label), (test_semantic, test_kg, test_label)


def transfer_torch(e, e_t, r, r_t):
    #TODO
    e_proj = t.sum(e * e_t) * r_t + e
    return e_proj + r


def process_data_into_tensor(data, label, tokenizer, entity2id, relation2id, split_flag, post_num, post_length):
    user_names = data.id.values
    posts = data.posts.values
    labels = data.loc[:, [label]].values.squeeze(1)

    data = []
    for post in tqdm(posts):
        user_post = post.split(split_flag)
        user_post = [p.strip() for p in user_post if len(p.strip()) != 0]
        while len(user_post) < post_num:
            user_post.append('')
        data.append(user_post[:post_num])

    user_ids = []
    relation_ids = []
    #TODO
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

    for user_name in tqdm(user_names):
        user_ids.append(entity2id[user_name])
        relation_ids.append(relationId)

    input_ids, attention_masks = [], []
    for user in tqdm(data):
        user_input_ids, user_attention_masks = [], []
        for post in user:
            encoded_dict = tokenizer.encode_plus(post, add_special_tokens=True, max_length=post_length,
                                                 pad_to_max_length=True, return_attention_mask=True)
            user_input_ids.append(encoded_dict['input_ids'])
            user_attention_masks.append(encoded_dict['attention_mask'])
        input_ids.append(user_input_ids)
        attention_masks.append(user_attention_masks)

    input_ids = t.tensor(input_ids)
    masks = t.tensor(attention_masks)
    user_ids = t.tensor(user_ids)
    relation_ids = t.tensor(relation_ids)
    labels = t.tensor(labels).float()

    return input_ids, masks, user_ids, relation_ids, labels


def get_trainset_for_version_0(file_path, inpath, dataset, embedding_size, train_rate, split_rate, version):
    '''
    针对版本0（加入用户文本相似关系）的图谱进行特征与标签的提取，为下一步训练提供数据集
    :param file_path: './dataset/*_for_kg.csv'
    :param inpath:储存entity2id的文件夹
    :param dataset:
    :param embedding_size:
    :param train_rate:
    :param split_rate:
    :param version:
    :return:[n * (embedding_size + 5)]
    '''
    data = pd.read_csv(file_path)
    user_ids = data.id.values
    labels = data.loc[:, ['ope', 'con', 'ext', 'agr', 'neu']].values

    entity2id, relation2id = get_hashmap(inpath)

    ent_embedding, rel_embedding, ent_transfer, rel_transfer = \
        get_entity_embedding(inpath, dataset, embedding_size, train_rate, split_rate, version)

    user_embedding_index = [entity2id[user_ids[i]] for i in range(data.shape[0])]
    dataset = ent_embedding[user_embedding_index]

    dataset = np.concatenate((dataset, labels), axis=1)

    return dataset


def svr_for_version_0(dataset, file_name, embedding_size, train_rate, split_rate, version, test_size):
    file_path = 'dataset/{}_for_kg.csv'.format(dataset)
    inpath = 'kg_embedding_dataset/{}/'.format(file_name)
    label_list = ['ope', 'con', 'ext', 'agr', 'neu']

    data = get_trainset_for_version_0(file_path, inpath, dataset, embedding_size, train_rate, split_rate, version)
    data = data.astype(np.float64)

    score_times = 10
    count = 0

    results_mae, results_mse = [], []

    for _ in tqdm(range(score_times)):
        train, test = train_test_split(data, test_size=test_size, shuffle=True)
        x_train = train[:, :-5]
        y_train = train[:, -5:]
        x_test = test[:, :-5]
        y_test = test[:, -5:]
        result_mae, result_mse = regression_svr(x_train, y_train, x_test, y_test)
        results_mae.append(result_mae)
        results_mse.append(result_mse)

    all_results_mae, min_score_mae, max_score_mae, mean_score_mae = [], [], [], []

    all_results_mse, min_score_mse, max_score_mse, mean_score_mse = [], [], [], []

    results_mae = pd.DataFrame(results_mae)
    all_results_mae.append(results_mae)
    mean_score_mae.append(results_mae.mean())
    max_score_mae.append(results_mae.max())
    min_score_mae.append(results_mae.min())

    results_mse = pd.DataFrame(results_mse)
    all_results_mse.append(results_mse)
    mean_score_mse.append(results_mse.mean())
    max_score_mse.append(results_mse.max())
    min_score_mse.append(results_mse.min())

    print(label_list)
    print('MSE mean:\n')
    print(mean_score_mse)
    print('\n')
    print('MSE min:\n')
    print(min_score_mse)
    print('\n')
    print('MSE max:\n')
    print(max_score_mse)
    print('\n')

    print(label_list)
    print('MAE mean:\n')
    print(mean_score_mae)
    print('\n')
    print('MAE min:\n')
    print(min_score_mae)
    print('\n')
    print('MAE max:\n')
    print(max_score_mae)
    print('\n')


def regression_svr(train_x, train_label, test_x, test_label):
    clf = MultiOutputRegressor(svm.SVR(gamma='scale'))
    clf.fit(train_x, train_label)
    y_pred = clf.predict(test_x)

    mae = mean_absolute_error(test_label, y_pred, multioutput='raw_values')
    mse = mean_squared_error(test_label, y_pred, multioutput='raw_values')

    result_mae, result_mse = [], []

    for i in range(5):
        result_mae.append(mae[i])
        result_mse.append(mse[i])

    return result_mae, result_mse


'''
===========================================
NRL预测函数
'''


def get_trainset_for_NRL(file_path, inpath):
    """
    针对版本0（加入用户文本相似关系）的图谱进行特征与标签的提取，为下一步训练提供数据集
    :param file_path: './dataset/*_for_kg.csv'
    :param inpath:储存entity2id的文件夹
    :param dataset:
    :param embedding_size:
    :param train_rate:
    :param split_rate:
    :param version:
    :return:[n * (embedding_size + 5)]
    """
    data = pd.read_csv(file_path)
    user_ids = data.id.values
    labels = data.loc[:, ['ope', 'con', 'ext', 'agr', 'neu']].values

    model = word2vec.Word2Vec.load('./NRL/youtube_11.model')

    entity2id, relation2id = get_hashmap(inpath)

    dataset = None

    for i in range(user_ids.shape[0]):
        user_id = str(entity2id(user_ids[i]))
        if dataset is None:
            dataset = np.expand_dims(model[user_id], 0)
        else:
            temp = np.expand_dims(model[user_id], 0)
            dataset = np.concatenate((dataset, temp), 1)
    dataset = np.concatenate((dataset, labels), axis=1)
    return dataset


def svr_for_NRL(dataset, file_name, embedding_size, train_rate, split_rate, version, test_size):
    file_path = 'dataset/{}_for_kg.csv'.format(dataset)
    inpath = 'kg_embedding_dataset/{}/'.format(file_name)
    label_list = ['ope', 'con', 'ext', 'agr', 'neu']

    data = get_trainset_for_version_0(file_path, inpath, dataset, embedding_size, train_rate, split_rate, version)
    data = data.astype(np.float64)

    score_times = 10
    count = 0

    results_mae, results_mse = [], []

    for _ in tqdm(range(score_times)):
        train, test = train_test_split(data, test_size=test_size, shuffle=True)
        x_train = train[:, :-5]
        y_train = train[:, -5:]
        x_test = test[:, :-5]
        y_test = test[:, -5:]
        result_mae, result_mse = regression_svr(x_train, y_train, x_test, y_test)
        results_mae.append(result_mae)
        results_mse.append(result_mse)

    all_results_mae, min_score_mae, max_score_mae, mean_score_mae = [], [], [], []

    all_results_mse, min_score_mse, max_score_mse, mean_score_mse = [], [], [], []

    results_mae = pd.DataFrame(results_mae)
    all_results_mae.append(results_mae)
    mean_score_mae.append(results_mae.mean())
    max_score_mae.append(results_mae.max())
    min_score_mae.append(results_mae.min())

    results_mse = pd.DataFrame(results_mse)
    all_results_mse.append(results_mse)
    mean_score_mse.append(results_mse.mean())
    max_score_mse.append(results_mse.max())
    min_score_mse.append(results_mse.min())

    print(label_list)
    print('MSE mean:\n')
    print(mean_score_mse)
    print('\n')
    print('MSE min:\n')
    print(min_score_mse)
    print('\n')
    print('MSE max:\n')
    print(max_score_mse)
    print('\n')

    print(label_list)
    print('MAE mean:\n')
    print(mean_score_mae)
    print('\n')
    print('MAE min:\n')
    print(min_score_mae)
    print('\n')
    print('MAE max:\n')
    print(max_score_mae)
    print('\n')
