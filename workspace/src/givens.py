import pandas as pd
import nltk

def loadDF(dataset):
    '''
    You will use this function to load the dataset into a Pandas Dataframe for processing.
    
    '''
    df = {
        "question": [],
        "answer": []
    }
    
    for context, question, answers, indices in dataset:
        if answers[0]:
            df["question"].append(question)
            df["answer"].append(answers[0])
        
        
    return pd.DataFrame.from_dict(df)




def prepare_text(sentence):
    
    '''

    Our text needs to be cleaned with a tokenizer. This function will perform that task.
    https://www.nltk.org/api/nltk.tokenize.html

    '''
    
    return nltk.tokenize.word_tokenize(sentence)




def train_test_split(SRC, TRG):
    
    '''
    Input: SRC, our list of questions from the dataset, should be a tuple of (train_df, test_df)
            TRG, our list of responses from the dataset

    Output: Training and test datasets for SRC & TRG

    '''
    SRC_train_dataset = train_df["question"].tolist()
    TRG_train_dataset = train_df["answer"].tolist()
    
    SRC_test_dataset = test_df["question"].tolist()
    TRG_test_dataset = test_df["answer"].tolist()
    
    
    return SRC_train_dataset, SRC_test_dataset, TRG_train_dataset, TRG_test_dataset




