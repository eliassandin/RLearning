# Project-in-Computer-Science
In the folder 'my_version' you'll find all the code i've written, plus all the models i've trained and saved.
The executing file is 'script.py'. From there, there are functions to load models, initialize players, play games and train.

If you run the command "$ python script.py" in terminal, a game will be played and shown between a policy player and a filter player. After that, 500 games will be played between them and the statistics will be shown to you. Right now Policy player wins 472/500 games against filter player.
The commands executed from 'script.py' are at line 225-269. There are different commands you can uncomment and try if you like.

In lines 111-165 in script.py you can see the reinforcement training. It uses functions from the file 'ModelFunctions.py' where i've stored all my training functions. Most recently the function 'train_batch' is used (line 88- 151 in 'ModelFunctions.py')

In the file Agent.py you can find all the different Player classes incl SmartPlayer4 which uses policy net, SmartPlayer3 which uses value net, filter player and most recently MCTSPlayer.

The different architectures i've tried are stored in NeuralModel.py. Most recently used is PolicyModel3 and CNNModel6 for policy and value net.

In test_random.py there are files for playing games.

connect_4_env is code i downloaded from git.
