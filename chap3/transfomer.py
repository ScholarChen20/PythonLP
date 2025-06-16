import torch
from torch.nn.functional import softmax
X=[
    [1,0,1,0],
    [0,2,0,2],
    [1,1,1,1]
]
x=torch.tensor(X,dtype=torch.float32)
# print(x)

w_key = [
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 0],
  [1, 1, 0]
]
w_query = [
  [1, 0, 1],
  [1, 0, 0],
  [0, 0, 1],
  [0, 1, 1]
]
w_value = [
  [0, 2, 0],
  [0, 3, 0],
  [1, 0, 3],
  [1, 1, 0]
]
w_key = torch.tensor(w_key, dtype=torch.float32)
w_query = torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)

# print("Weights for key: \n", w_key)
# print("Weights for query: \n", w_query)
# print("Weights for value: \n", w_value)

keys=x@w_key
querys=x@w_query
values=x@w_value
# print(keys)
# print(querys)
# print(values)

atten_scores=querys@keys
# print(atten_scores)

atten_scores_softmax=softmax(atten_scores,dim=-1)
# print(atten_scores_softmax)

atten_scores_softmax=[
    [0.0,0.5,0.5],[0.0,0.0,0.0],[0.0,0.9,0.1]
]
atten_scores_softmax=torch.tensor(atten_scores_softmax)
# print(atten_scores_softmax)

weighted_values=values[:,None]*atten_scores_softmax.T[:,:,None]
# print(weighted_values)
weight1=weighted_values[0]
weight2=weighted_values[1]
weight3=weighted_values[2]
for i in range(4):
    weight=weight1+weight2+weight3
print(weight)
