import Roberta_train
import load
import Get_Models
import train

td,tt,dd,dt,tl,dl = load.get_data('.',2)

rm,t,rc = Get_Models.get_roberta_model()

train.encoder(rm,t,rc,tl)
