# Encounter uknown values

To encounter uknown values a new branch "other" was entroduced with the child leaf node. The label of that child node is equal to most common value among the target attributes. The accuracy achieved by this is 64% which is not enough. The problem here is that if the uknown value is encounted in the top of the tree, then all other values of know attributes are ignored.


The outcome can also change from the training set to the real set, bcs new attributes come into the scene

# Attempt 2, Gain Ration measure