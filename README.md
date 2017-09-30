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

The username of the bot on Telegram is `@cip_nlp_chatbot`.

## How to Train the networks
To train the networks it's necessary to specify the structure of them with a JSON
file, for example:
~~~~
python3 train.py ../models/hparams.json
~~~~
Note that you need to have both `Keras` and `PyTorch`. Moreover, to train only
some of the networks you need to modify the flags `TRAIN_*` from the code.
The models will be saved in the folder `models`.

## How to Download the Knowledge Base
To download the knowledge base simply type from the terminal:
~~~~
python3 download_kb.py
~~~~
The knowledge base will be saved on a JSON file in the folder `resources`.

## How to Generate a Vocabulary
To generate a vocabulary from the knowledge base simply type from the terminal:
~~~~
python3 vocabulary_from_kb.py N
~~~~
where `N` is the minimum number of times a word has to appear in the knowledge
base to be included in the vocabulary. The vocabulary will be saved in the
folder `resources`.

## How to Generate Question Patterns File
To generate a unique question pattern file from multiple files type from the terminal:
~~~~
python3 patterns.py
~~~~
Note that before running this script you have to remove bad files and after the
execution of the script you have to double check the file created in the
folder `resources` by hand.

## How to Query the Knowledge Base
To debug the bot a simple script has been developed to search for words or to
select a random tuple randomly. Run the script by typing:
~~~~
python3 query_kb.py
~~~~
Once the program has been loaded the knowledge base type `search=SENTENCE` to
search for a sentence, `random` to randomly select a tuple or `quit` to exit
from the program.

## Dependencies
* Keras (built on top of Tensorflow)
* PyTorch
* Telepot
* Gensim
