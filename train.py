import load_data
import roberta_training
import torch

device = torch.device("cuda")

def encoder(roberta_model,roberta_tokenizer,roberta_config,train_loader):
    roberta_model.train()

    count = 0
    i=0

    for batch_index, batch in enumerate((train_loader)):
        if i:
            break
        count += len(batch)


        # nlu  : natural language utterance
        # nlu_t: tokenized nlu
        # sql_i: canonical form of SQL query
        # sql_q: full SQL query text. Not used.
        # sql_t: tokenized SQL query
        # tb   : table metadata. No row data needed
        # hs_t : tokenized headers. Not used.
        natural_lang_utterance, natural_lang_utterance_tokenized, sql_canonical, \
            _, _, table_metadata, _, headers = load_data.get_fields(batch)


        # select_column_ground, select_agg_ground, where_number_ground, \
        #     where_column_ground, where_operator_ground, _ = roberta_training.get_ground_truth_values(sql_canonical)
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        

        natural_lang_embeddings, header_embeddings, question_token_length, header_token_length, header_count, \
        natural_lang_double_tokenized, punkt_to_roberta_token_indices, roberta_to_punkt_token_indices \
            = roberta_training.get_wemb_roberta(roberta_config, roberta_model, roberta_tokenizer, 
                                        natural_lang_utterance_tokenized, headers,max_seq_length= 222,
                                        num_out_layers_n=2, num_out_layers_h=2)
        print("utterance",natural_lang_utterance)
        print("headers",headers)

        print("embeddings: ", natural_lang_embeddings)
        print("header embeddings",header_embeddings)
        print("question token len",question_token_length)
        print("header token len",header_token_length)
        print("header count",header_count)
        print("double", natural_lang_double_tokenized)
        print("punkt",punkt_to_roberta_token_indices)
        print("roberta punkt",roberta_to_punkt_token_indices)

        i+=1


# def train(seq2sql_model,roberta_model,model_optimizer,roberta_optimizer,roberta_tokenizer,roberta_config,path_wikisql,train_loader):

#     roberta_model.train()
#     seq2sql_model.train()
    
#     results=[]
#     average_loss = 0
    
#     count_select_column = 0  # count the # of correct predictions of select column
#     count_select_agg = 0  # of selectd aggregation
#     count_where_number = 0  # of where number
#     count_where_column = 0  # of where column
#     count_where_operator = 0  # of where operator
#     count_where_value = 0  # of where-value
#     count_where_value_index = 0  # of where-value index (on question tokens)
#     count_logical_form_acc = 0  # of logical form accuracy
#     count_execution_acc = 0  # of execution accuracy


#     # Engine for SQL querying.
#     #engine = DBEngine(os.path.join(path_wikisql, f"train.db"))
#     count = 0  # count the # of examples
#     for batch_index, batch in enumerate(tqdm(train_loader)):
#         count += len(batch)

#         # if batch_index > 2:
#         #     break
#         # Get fields

#         # nlu  : natural language utterance
#         # nlu_t: tokenized nlu
#         # sql_i: canonical form of SQL query
#         # sql_q: full SQL query text. Not used.
#         # sql_t: tokenized SQL query
#         # tb   : table metadata. No row data needed
#         # hs_t : tokenized headers. Not used.
#         natural_lang_utterance, natural_lang_utterance_tokenized, sql_canonical, \
#             _, _, table_metadata, _, headers = load_data.get_fields(batch)


#         select_column_ground, select_agg_ground, where_number_ground, \
#             where_column_ground, where_operator_ground, _ = roberta_training.get_ground_truth_values(sql_canonical)
#         # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        

#         natural_lang_embeddings, header_embeddings, question_token_length, header_token_length, header_count, \
#         natural_lang_double_tokenized, punkt_to_roberta_token_indices, roberta_to_punkt_token_indices \
#             = roberta_training.get_wemb_roberta(roberta_config, roberta_model, roberta_tokenizer, 
#                                         natural_lang_utterance_tokenized, headers,max_seq_length= 222,
#                                         num_out_layers_n=2, num_out_layers_h=2)
                                    
