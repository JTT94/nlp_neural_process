
import pandas as pd
import os
from neural_process.params import NeuralProcessParams
from neural_process.neural_process import NLP_NeuralProcess
from datetime import datetime
## For kaggle dataset


## Load data
data_dir = r'C:\Users\james\Documents\Oxford\Jigsaw\kaggle_toxic_comments'
filename = os.path.join(data_dir, 'train.csv')
df = pd.read_csv(filename)
cols = ['comment_text','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df = df[cols]

score_column = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

text_col_name = 'comment_text'

# Cast to float - because scores and labels need to be concattenated in the model function, and so need to be same type
for i in score_column:
  df[i] = pd.to_numeric(df[i],downcast='float')


# Init NP

np_params = NeuralProcessParams(dim_z=1000, n_hidden_units_h=[512, 256, 128], n_hidden_units_g=[512, 256, 128])
num_train_steps = 1

neural_process = NLP_NeuralProcess(score_col=score_column, params = np_params, num_draws = 25,
                                   num_train_steps=num_train_steps,
                                   num_warmup_steps=1,
                                   num_classes = len(score_column),
                                   output_dir = './',
                                  context_features = None)


df_train = df
print(datetime.now())
neural_process.train(df_train=df_train,
                                  num_train_steps=num_train_steps,
                                  score_col= score_column,
                                  text_col=text_col_name,)
print(datetime.now())