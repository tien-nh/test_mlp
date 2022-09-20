
import json
from supervisor import MLPSupervisor

problem_config = json.load(open("problem_config.json", "r", encoding="utf8"))
test_config = json.load(open("test.json", "r", encoding="utf8"))

sup = MLPSupervisor(problem_config) 
sup.train()
sup.test(test_config)