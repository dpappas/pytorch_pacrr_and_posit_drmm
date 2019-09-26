
from py2neo import Graph, Node
graph = Graph(password="pass")


alice = Node("Person", name="Alice")
graph.create(alice)
german, speaks = graph.create({"name": "German"}, (alice, "SPEAKS", 0))






