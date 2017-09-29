# NLP-Chatbot

## How to Run the Bot
The bot can be launched using different algorithms to generate answers and specifying
how the bot will be used. There are three algorithms that can be used
to specify the answer: `--answergenerator`, `--conceptextractor`, `--seq2seq`. There are
three ways to use the bot: `--onlyquestion`, `--onlyanswer`, `--bothQA`.
* `--answergenerator`: doesn't make any use of learning algorithms;
* `--conceptextractor`: uses two neural networks, one to classify the relation of the
question and another one to extract concepts;
* `--seq2seq`: uses Seq2Seq algorithm (end-to-end approach).
* `--onlyquestion`: the user can only ask questions (querying);
* `--onlyanswer`: the user can only answer to the questions of the bot (enriching);
* `--bothQA`: both querying and enriching.
To run the bot it's necessary to specify a JSON file that describes the network, the
algorithm to answer to the questions and the way the bot will be used. For example,
to run the bot that does both querying and enriching with concept extrator algorithm,
it's necessary to type:
~~~~
python3 bot.py ../models/hparams.json --conceptextractor --bothQA
~~~~

## How to Train the networks
How to train the networks.

## How to Download the Knowledge Base
How to dowload the kb.

## How to Generate a Vocabulary
How to generate a vocabulary.

## How to Generate Question Patterns File
How to generate questions patterns.

## How to Query the Knowledge Base
How to query the knowledge base.
