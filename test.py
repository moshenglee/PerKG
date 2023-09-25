import tools
import pickle
# from SEMANTIC import bert_train as semantic_tool
import tools_plus

embedding_size = 200
train_rate = 0.5
split_rate = 0.2
version = 1

dataset = 'youtube'
file = 'youtube_0.5'

'''
=============
NRL实验步骤 版本0
'''
# tools.delete_user_similar(dataset)
# tools.delete_user(dataset)
#
# tools.add_user_linguistic(dataset)
# tools.add_user_similarity(dataset, 0.05)
#
# call apoc.export.json.all('youtube_2.json', {useTypes:true})
#
tools.process_kg_json('kg_json/{}.json'.format(file), 'kg_embedding_dataset/{}'.format(file))
# 接下来在NRL_test.py中进行实验

'''
=============
'''

'''
一系列图谱操作 版本1
'''

# tools.delete_user_similar(dataset)
# tools.delete_user(dataset)
# tools.delete_user_personality(dataset)
#
# tools.add_user_linguistic(dataset)
# tools.add_user_personality(dataset, train_rate, split_rate, version)
# tools.add_user_similarity(dataset, 0.05)

# 版本2弃用，即加入dividedinto 关系的
# tools.change_kg_from_1_to_2()
# tools.change_kg_from_2_to_1()

# call apoc.export.json.all('youtube_2.json', {useTypes:true})

# 处理JSON文件
# tools.process_kg_json('kg_json/{}.json'.format(file), 'kg_embedding_dataset/{}'.format(file))
# tools.process_kg_json('kg_json/my_2.json', 'kg_embedding_dataset/{}'.format(file))

# tools.train('kg_embedding_dataset/{}/'.format(file), dataset, embedding_size, train_rate, split_rate, version)

# kg_mae = tools.score_SVR('MAE', dataset, file, embedding_size, 1 - train_rate, train_rate, split_rate, version)

# kg_mae = tools.single_score_SVR('MAE', dataset, file, embedding_size, train_rate, split_rate, version)

# kg_mse, semantic_mse, both_mse, kg_mae, semantic_mae, both_mae = \
#     tools_plus.single_score_SVR('MAE', dataset, file, embedding_size, train_rate, split_rate, version, 0, 100)


# kg_mse, semantic_mse, both_mse, kg_mae, semantic_mae, both_mae = \
#     tools_plus.single_score_MLP('MAE', dataset, file, embedding_size, train_rate, split_rate, version, 0, 100)

# print('MSE\n')
# print('使用图谱向量预测结果：\n')
# print(kg_mse)
# print('\n')
# print('使用语义向量预测结果：\n')
# print(semantic_mse)
# print('\n')
# print('使用结合向量预测结果：\n')
# print(both_mse)
#
# print('MAE\n')
# print('使用图谱向量预测结果：\n')
# print(kg_mae)
# print('\n')
# print('使用语义向量预测结果：\n')
# print(semantic_mae)
# print('\n')
# print('使用结合向量预测结果：\n')
# print(both_mae)

# tools_plus.get_features_specified_label('kg_embedding_dataset/{}/'.format(file), 'my', 'my_1',
#                                         200, 'ope', 0.833, 0.3, 1, 0, 100)


'''
====================================================
分割线，下边为深度学习模型部分得训练与测试代码
====================================================
'''
# semantic_tool.train('my', 'ope', 20, 8, 32, 16, 5, 0, file)
# semantic_tool.train('my', 'con', 20, 8, 32, 16, 5, 0, file)
# semantic_tool.train('my', 'agr', 20, 8, 32, 16, 5, 1, file)
# semantic_tool.train('my', 'ext', 20, 8, 32, 16, 5, 2, file)
# semantic_tool.train('my', 'neu', 20, 8, 32, 16, 5, 3, file)


# result = {}
# result['ope'] = semantic_tool.test('my', 'ope', 8, 32, 16, 0, file)
#
# result['con'] = semantic_tool.test('my', 'con', 8, 32, 16, 0, file)
#
# result['ext'] = semantic_tool.test('my', 'ext', 8, 32, 16, 0, file)
#
# result['agr'] = semantic_tool.test('my', 'agr', 8, 32, 16, 0, file)
#
# result['neu'] = semantic_tool.test('my', 'neu', 8, 32, 16, 0, file)
#
# print(result)

'''
====================================================
分割线，下边为加入语义相似关系的图谱测试部分
====================================================
'''

# tools_plus.svr_for_version_0(dataset, file, embedding_size, train_rate, split_rate, version, 0.5)
