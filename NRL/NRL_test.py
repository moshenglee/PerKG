import networkx as nx
import numpy as np
import walk
import re
from gensim.models import word2vec
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

file = 'youtube_0'


# dataset = 'dataset'


def get_entity_type(entity):
    """
    根据实体名称判断实体类型
    :param entity:
    :return:
    """
    pattern_user = 'VLOG(\d+)'
    pattern_word = '.*(\+|-)'
    if re.search(pattern_user, entity) is not None:
        return 'USER'
    if re.search(pattern_word, entity) is not None:
        return 'WORD'
    return 'PERSONALITY'


def generate_graph(file):
    """
    根据实验名称，获取需要的graph, entity2id, id2entity
    :param file:
    :return:
    """
    G = nx.Graph()

    id2entity = {}
    entity2id = {}

    # 读取entity2id文件
    with open('../kg_embedding_dataset/{}/entity2id.txt'.format(file)) as f:
        entity_num = int(f.readline())
        for line in f.readlines():
            line = line.split()
            id2entity[int(line[1])] = line[0]
            entity2id[line[0]] = int(line[1])

    print('实体数量为：{}'.format(entity_num))

    # 初始化邻接矩阵
    # 对于与其他节点均无关系的节点，做嵌入时直接跳过
    adjacent = np.identity(entity_num)

    # 读取triple2id文件，构造邻接矩阵
    with open('../kg_embedding_dataset/{}/triple2id.txt'.format(file)) as f:
        triple_num = int(f.readline())
        for line in f.readlines():
            triple = line.split()

            G.add_node(int(triple[0]), entity_type=get_entity_type(id2entity[int(triple[0])]))
            G.add_node(int(triple[1]), entity_type=get_entity_type(id2entity[int(triple[1])]))

            if len(triple) == 4:
                h, t, r, s = int(triple[0]), int(triple[1]), int(triple[2]), float(triple[3])
                adjacent[h][t] = 1
                adjacent[t][h] = 1
                G.add_edge(h, t, rel=r, sim=s)
            else:
                h, t, r = int(triple[0]), int(triple[1]), int(triple[2])
                adjacent[h][t] = 1
                adjacent[t][h] = 1
                G.add_edge(h, t, rel=r)

    # 矩阵形状为578 * 578
    # G中节点数为519（其余为单独节点）
    # 28471条关系（无向关系）
    print(len(G.nodes))
    print(len(G.edges))

    return G, entity2id, id2entity


# for i, node in enumerate(G.nodes(data=True)):
#
#     print('第 {} 节点'.format(i))
#     node_id = node[0]
#     node_type = node[1]['entity_type']
#
#     print('node_id : {} node_type : {} 邻居节点数量 : {}'.format(id2entity[node_id], node_type, len(sorted(G.neighbors(node_id)))))
#     if len(sorted(G.neighbors(node_id))) == 1:
#         print(id2entity[sorted(G.neighbors(node_id))[0]])

def NRL_word2vec(file, p, q, pf, qf):
    """
    网络表示学习
    通过游走策略生成路径，基于word2vec训练并保存模型
    :param file:
    :return:
    """
    g = walk.Graph(G, 0, p, q, pf, qf)
    print('构建游走图！')
    g.preprocess_trainsition_probs()
    print('预处理矩阵完成！')
    paths = g.simulate_walks(30, 80)

    paths_str = [[str(node) for node in path] for path in paths]

    print(len(paths_str))

    model = word2vec.Word2Vec(paths_str)
    model.save('model_save/{}.model'.format(file))


def get_trainset_for_NRL(file, file_path, entity2id):
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

    model = word2vec.Word2Vec.load('model_save/{}.model'.format(file))

    dataset = None

    for i in range(user_ids.shape[0]):
        user_id = str(entity2id[user_ids[i]])
        if dataset is None:
            dataset = np.expand_dims(model.wv[user_id], 0)
        else:
            temp = np.expand_dims(model.wv[user_id], 0)
            dataset = np.concatenate((dataset, temp), 0)
    print(dataset.shape)
    print(labels.shape)
    dataset = np.concatenate((dataset, labels), axis=1)
    return dataset


def regression_svr(train_x, train_label, test_x, test_label):
    clf = MultiOutputRegressor(svm.SVR(gamma='scale'))
    clf.fit(train_x, train_label)
    y_pred = clf.predict(test_x)

    mae = mean_absolute_error(test_label, y_pred, multioutput='raw_values')
    # mse = mean_squared_error(test_label, y_pred, multioutput='raw_values')

    # result_mae, result_mse = [], []
    result_mae = []
    for i in range(5):
        result_mae.append(mae[i])
        # result_mse.append(mse[i])

    return result_mae
    # , result_mse


def svr_for_NRL(dataset, file, entity2id, test_size):
    file_path = '../../datasets/{}_for_kg.csv'.format(dataset)
    label_list = ['ope', 'con', 'ext', 'agr', 'neu']

    data = get_trainset_for_NRL(file, file_path, entity2id)
    data = data.astype(np.float64)

    score_times = 10
    count = 0

    # results_mae, results_mse = [], []
    results_mae = []

    for _ in tqdm(range(score_times)):
        train, test = train_test_split(data, test_size=test_size, shuffle=True)
        x_train = train[:, :-5]
        y_train = train[:, -5:]
        x_test = test[:, :-5]
        y_test = test[:, -5:]
        result_mae = regression_svr(x_train, y_train, x_test, y_test)
        results_mae.append(result_mae)
        # results_mse.append(result_mse)

    all_results_mae, min_score_mae, max_score_mae, mean_score_mae = [], [], [], []

    # all_results_mse, min_score_mse, max_score_mse, mean_score_mse = [], [], [], []

    results_mae = pd.DataFrame(results_mae)
    all_results_mae.append(results_mae)
    mean_score_mae.append(results_mae.mean())
    # max_score_mae.append(results_mae.max())
    # min_score_mae.append(results_mae.min())

    # results_mse = pd.DataFrame(results_mse)
    # all_results_mse.append(results_mse)
    # mean_score_mse.append(results_mse.mean())
    # max_score_mse.append(results_mse.max())
    # min_score_mse.append(results_mse.min())

    # print(label_list)
    # print('MSE mean:\n')
    # print(results_mse.mean().values)
    # print('\n')
    # print('MSE min:\n')
    # print(results_mse.min().values)
    # print('\n')
    # print('MSE max:\n')
    # print(results_mse.max().values)
    # print('\n')

    # print(label_list)
    print('test size:', test_size)
    print('MAE mean:')
    print(results_mae.mean().values)
    # print('\n')
    # print('MAE min:\n')
    # print(results_mae.min().values)
    # print('\n')
    # print('MAE max:\n')
    # print(results_mae.max().values)
    # print('\n')


# file_path = '../dataset/{}_for_kg.csv'.format('youtube')

G, entity2id, id2entity = generate_graph(file)
NRL_word2vec(file, 1, 1, 5, 1)

# for i in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
svr_for_NRL('youtube', file, entity2id, 0.5)


