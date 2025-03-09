import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Dataloader
import torchtext
from torchtext.vocab import GloVe
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pand import problems_df, solutions_df

data_df = pd.concat([problems_df, solutions_df], axis=1)

#Creating a custom dataset class
class ProblemDataset(Dataset):
    def __init__(self, data_df, vocab):
        self.data_df = data_df
        self.vocab = vocab

    def __getitem__(self, index):
        problem = self.data_df.iloc[index, 0]
        solution = self.data_df.iloc[index, 1]
        
        # Preprocess the problem and solution
        problem_tokens = word_tokenize(problem)
        solution_tokens = word_tokenize(solution)

        #Conver the tokens to numerical tensors
        problem_tensor = torch.tensor([self.vocab[token] for token in problem_tokens])
        solution_tensor = torch.tensor([self.vocab[token] for token in solution_tokens])

        return problem_tensor, solution_tensor
    
    def __len__(self):
        return len(self.data_df)
    
#Creating a vocabulary
vocab = GloVe(name = '6B', dim=100)

#Create a dataset Interface
dataset = ProblemDataset(data_df, vocab)

data_loader = Dataloader(dataset, batch_size=32, shuffle=True)

#
class ProblemModel(nn.Module):
    def __init__(self):
        super(ProblemModel, self).__int__()
        self.embedding = nn.Embedding(len(vocab),100)
        self.rnn = nn.GRU(input_size=100, hidden_size=128, num_layers=1, batch_firs=True)
        self.fc = nn.Linear(128, len(vocab))

model = ProblemModel()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_fn = nn.CrossEntropyLoss()


#Model Training
for epoch in range(10):
    for batch in data_loader:
        problem_tensor, solution_tensor = batch 
        optimizer.zero_grad()
        output = model(problem_tensor)
        loss = loss_fn(output, torch.arg,max(solution_tensor, dim=1))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

#Using model for interface
def get_solution(problem):
    problem_tokens = word_tokenize(problem)
    problem_tensor = torch.tensor([vocab[token] for token in problem_tokens])
    output = model(problem_tensor)
    solution_index = torch.argmax(output)
    solution_token =vocab.itos[solution_index]
    return solution_token

if __name__ == '__main__':
    problem = 'I am feeling sad'
    print(get_solution(problem))
