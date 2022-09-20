
import json
from supervisor import En_DecoderSupervisor

problem_config = json.load(open("problem_config_lstm.json", "r", encoding="utf8"))
test_config = json.load(open("test.json", "r", encoding="utf8"))

sup = En_DecoderSupervisor(problem_config) 
sup.train()
sup.test(test_config)