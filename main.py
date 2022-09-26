import roberta_training
import load_data
import load_model
import train

td,tt,dd,dt,tl,dl = load_data.get_data('.',2)

rm,t,rc = load_model.get_roberta_model()

train.encoder(rm,t,rc,tl)
