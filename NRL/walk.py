import networkx as nx
import numpy as np
import random
from tqdm import tqdm


class Graph():
    def __init__(self, nx_G, is_directed, p, q, pf, qf):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.pf = pf
        self.qf = qf

    def walk(self, walk_length, start_node):
        '''
        从一个初始节点计算一个随机游走
        :param walk_length: 随机游走序列长度
        :param start_node: 初始节点
        :return: 列表，随机游走序列
        '''
        G = self.G

        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break
        # logging.info(str(start_node) + 'random walk end...')
        return walk

    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        walks = []
        # logging.info('all nodes to list')
        nodes = list(G.nodes())
        # logging.info('walk iteration:')
        for walk_iter in tqdm(range(num_walks)):
            # logging.info(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.walk(walk_length=walk_length, start_node=node))
        # logging.info('walk iteration end')
        return walks

    def get_alias_edge(self, src, dst):
        G = self.G
        p = self.p
        q = self.q
        pf = self.pf
        qf = self.qf
        K1 = G.degree(src) / G.number_of_nodes()

        unnormalized_probs = []

        '''
        ==========================================================
        此处为新的游走策略
        '''
        nodes = G.nodes(data=True)

        # 对于前一个节点和当前节点均为user的情况进行处理,下一步节点可能为user或者word
        # 整体选择基于node2vec进行，使用p, q参数对深度优先和广度优先的策略进行选择，同时，对于用户之间的概率
        # 均使用1 + similarity作为基准，整体上游走策略倾向于选择user多于word
        # 1.对于下一步返回前一节点的情况，下一节点一定也是user，因此概率为(1 + G[dst][dst_nbr]['sim']) / p
        # 2.对于下一步节点与前一节点距离为1的情况（bfs），分为user和word两种情况处理
        #   user:1 + G[dst][dst_nbr]['sim']
        #   word:1
        # 3.对于下一步节点与前一节点距离为2的情况(dfs)，同样分为user和word两种情况处理
        #   user:(1 + G[dst][dst_nbr]['sim']) / q
        #   word:1 / q
        if nodes[src]['entity_type'] == 'USER' and nodes[dst]['entity_type'] == 'USER':
            for dst_nbr in sorted(G.neighbors(dst)):
                # 下一节点返回前一节点，也为user
                if dst_nbr == src and nodes[dst_nbr]['entity_type'] == 'USER':
                    unnormalized_probs.append((1 + G[dst][dst_nbr]['sim']) / p)
                # 下一节点为前一节点的邻居节点，分两种情况处理，user和word
                # user: 1+两个用户间的文本相似度
                # word: 1
                elif G.has_edge(dst_nbr, src):
                    if nodes[dst_nbr]['entity_type'] == 'USER':
                        unnormalized_probs.append(1 + G[dst][dst_nbr]['sim'])
                    else:
                        unnormalized_probs.append(1)
                else:
                    if nodes[dst_nbr]['entity_type'] == 'USER':
                        unnormalized_probs.append((1 + G[dst][dst_nbr]['sim']) / q)
                    else:
                        unnormalized_probs.append(1 / q)
        # 对于前一个节点为word和当前节点为user的情况进行处理,下一步节点可能为user或者word
        # 整体选择基于node2vec进行，使用p, q参数对深度优先和广度优先的策略进行选择，同时，对于用户之间的概率
        # 均使用1 + similarity作为基准，整体上游走策略倾向于选择user多于word
        # 1.对于下一步返回前一节点的情况，下一节点一定也是word，因此概率为1 / p
        # 2.对于下一步节点与前一节点距离为1的情况（bfs），分为user和word两种情况处理
        #   user:1 + G[dst][dst_nbr]['sim']
        #   word:1
        # 3.对于下一步节点与前一节点距离为2的情况(dfs)，同样分为user和word两种情况处理
        #   user:(1 + G[dst][dst_nbr]['sim']) / q
        #   word:1 / q
        elif nodes[src]['entity_type'] == 'WORD' and nodes[dst]['entity_type'] == 'USER':
            for dst_nbr in sorted(G.neighbors(dst)):
                # 下一节点返回前一节点，也为user
                if dst_nbr == src and nodes[dst_nbr]['entity_type'] == 'WORD':
                    unnormalized_probs.append(1 / p)
                # 下一节点为前一节点的邻居节点，分两种情况处理，user和word
                # user: 1+两个用户间的文本相似度
                # word: 1
                elif G.has_edge(dst_nbr, src):
                    if nodes[dst_nbr]['entity_type'] == 'USER':
                        unnormalized_probs.append(1 + G[dst][dst_nbr]['sim'])
                    else:
                        unnormalized_probs.append(1)
                else:
                    if nodes[dst_nbr]['entity_type'] == 'USER':
                        unnormalized_probs.append((1 + G[dst][dst_nbr]['sim']) / q)
                    else:
                        unnormalized_probs.append(1 / q)
        else:
            for dst_nbr in sorted(G.neighbors(dst)):
                if dst_nbr == src:
                    unnormalized_probs.append(1 / pf)
                elif G.has_edge(dst_nbr, src):
                    unnormalized_probs.append(1)
                else:
                    unnormalized_probs.append(1 / qf)

        '''
        ==========================================================
        '''

        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return alias_setup(normalized_probs)

    def preprocess_trainsition_probs(self):
        '''
        preprocessing of transition probabilities for guiding the random walks
        :return:
        '''
        # logging.info('start preprocessing of transition probabilities for guiding the random walks')
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        i = 0
        # logging.info('nodes build start...')
        for node in tqdm(G.nodes()):
            i = i + 1
            if i % 100 == 0:
                print(i, ' nodes have been built.')
                # logging.info(str(i) + ' nodes have been build')
            unnormalized_probs = [1.0 for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)
        # logging.info('nodes build end...')
        alias_edges = {}
        triads = {}
        # logging.info('edges build start...')
        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            j = 0
            print('edge number: ', G.number_of_edges())
            for edge in tqdm(G.edges()):
                j = j + 1
                # if j % 1000 == 0:
                # logging.info(str(j) + ' alias_edges have been build')
                # print(j, ' edges have been built.')
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
        # logging.info('edges build end...')
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        # logging.info('end --- preprocessing of transition probabilities for guiding the random walks')
        return


def alias_setup(probs):
    """
    根据二阶random walk输出的概率变成每个节点对应的两个数，被后面的alias_draw函数进行抽样
    :param probs:输入概率，得到对应的两个列表，一个是在原始的prob数组[0.4,0.8,0.6,1]另外就是在上面补充的alias数组
    其值代表填充的那一列的序号索引
    q:accept
    J:alias
    :return:
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []

    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        # prob是否大于均值 1 / K
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def alias_draw(J, q):
    """
    抽样函数，使用alias采样从一个非均匀离散分布中采样
    :param J: alias
    :param q: accept
    :return:
    """
    K = len(J)
    # 从整体均匀混合分布中采样
    kk = int(np.floor(np.random.rand() * K))
    # 从二元混合中采样，要么保留较小的，要么选择更大的
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
