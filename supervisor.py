import torch
from torch import Tensor
from dataset import get_data, get_set_and_loader
from mlp import MLP as Model
from tqdm import tqdm
from utils import indicator
import pandas as pd 



class Supervisor():
    def __init__(self, problem_config):
        # self.dataloader = Dataloader(problem_config)
        # self.number_source = self.dataloader.get_number_sources()
        # self.number_target = self.dataloader.get_number_targets()
        
        self.problem_config = problem_config
        # self.model_config = model_config
       
        self.epochs = problem_config['epochs']
       
        self.device = problem_config['device'] if torch.cuda.is_available() else 'cpu'

        self.model = Model(problem_config)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = torch.nn.MSELoss()
    
    def train(self): 
        print("{action:-^50}".format(action="Training"))
        self.data_train = get_data(self.problem_config["file_path"], self.problem_config)
        dataset, loader = get_set_and_loader(self.data_train["X"], self.data_train["Y"])
        self.model = self.model.to(self.device)
        for epoch in tqdm(range(self.epochs)):
            for x, y in loader : 
                # print(x.shape, y.shape)
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()

    def test(self, test_config): 
        print("{action:-^50}".format(action="Test"))
        data_test = get_data(self.problem_config["file_path"],test_config,self.data_train['x-scaler'],self.data_train['y-scaler'])
        leng = len(data_test["Y"])
        dataset, loader = get_set_and_loader(data_test["X"], data_test["Y"], batch_size = leng, shuffle=False)
        metrics = {}
        for x, y in loader: 
            with torch.no_grad():
                model = self.model.to('cpu')
                pred = model(x)

                scaler = self.data_train["y-scaler"]
                y = scaler.inverse_transform(y)
                pred = scaler.inverse_transform(pred)
                metrics['result'] = indicator(torch.tensor(pred), torch.tensor(y))
                results = pd.DataFrame.from_dict(metrics, orient='index')
                name_file_csv =  "results/metric_" + str(self.problem_config["p"]) + "_.csv"
                results.to_csv(name_file_csv)
                

    

                
