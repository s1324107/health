"""MealMate AI: train a Japanese-food classifier. Python 3.10+.
Images must be placed in data/train/<class> and data/test/<class>.
No images from test are used to fit parameters or select threshold.
"""
import argparse, csv, json, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent
LABELS = ['jelly', 'pudding', 'yogurt', 'ramen', 'curry_rice', 'onigiri', 'other']
JA = {'jelly':'ゼリー','pudding':'プリン','yogurt':'ヨーグルト','ramen':'ラーメン','curry_rice':'カレーライス','onigiri':'おにぎり','other':'その他'}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--epochs',type=int,default=12)
 ap.add_argument('--batch-size',type=int,default=16)
 ap.add_argument('--lr',type=float,default=0.001)
 ap.add_argument('--seed',type=int,default=42)
 args=ap.parse_args()
 random.seed(args.seed); torch.manual_seed(args.seed)
 train_dir=ROOT/'data'/'train'; test_dir=ROOT/'data'/'test'
 allowed={'.jpg','.jpeg','.png','.webp','.bmp'}
 counts={}
 for split in ('train','test'):
  counts[split]={c:sum(f.suffix.lower() in allowed for f in (ROOT/'data'/split/c).iterdir()) for c in LABELS}
  absent=[c for c,n in counts[split].items() if n==0]
  if absent:
   raise SystemExit(f'画像が足りません: data/{split}/ '+', '.join(absent)+'。各カテゴリに画像を入れてください。')
 print('Images:',counts)
 print('注意: 同じ商品の連写や類似画像をtrainとtestに分けないでください。')
 tr=transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224,scale=(.72,1.0)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(brightness=.18,contrast=.18),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
 te=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
 a=datasets.ImageFolder(train_dir,transform=tr)
 b=datasets.ImageFolder(test_dir,transform=te)
 if set(a.classes)!=set(LABELS) or a.classes!=b.classes: raise SystemExit('train/testフォルダのラベル名が一致しません。')
 device='cuda' if torch.cuda.is_available() else 'cpu'
 print('Device:',device,'Classes:',a.classes)
 # Pretrained ImageNet features, not a fine-tuned Japanese-food model yet.
 try: net=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
 except Exception as e: raise SystemExit('事前学習済みモデルをダウンロードできません。ネット接続を確認してください。 '+str(e))
 for p in net.features.parameters(): p.requires_grad=False
 net.classifier[3]=nn.Linear(net.classifier[3].in_features,len(a.classes))
 net=net.to(device)
 loader=DataLoader(a,batch_size=args.batch_size,shuffle=True,num_workers=0)
 optimizer=torch.optim.AdamW(net.classifier.parameters(),lr=args.lr,weight_decay=.01)
 criterion=nn.CrossEntropyLoss()
 for epoch in range(args.epochs):
  net.train(); total=correct=0; loss_sum=0.
  for x,y in loader:
   x,y=x.to(device),y.to(device); optimizer.zero_grad(); logits=net(x); loss=criterion(logits,y)
   loss.backward();optimizer.step();loss_sum+=loss.item()*len(y);total+=len(y);correct+=(logits.argmax(1)==y).sum().item()
  print(f'Epoch {epoch+1}/{args.epochs} train_loss={loss_sum/total:.4f} train_accuracy={correct/total:.3f}')
 # TEST set should only be used for final reporting, not tuning threshold.
 net.eval(); td=DataLoader(b,batch_size=16,shuffle=False,num_workers=0)
 rows=[]; correct=total=0; cm=[[0]*len(a.classes) for _ in a.classes]
 with torch.no_grad():
  idx=0
  for x,y in td:
   probs=net(x.to(device)).softmax(dim=1).cpu(); preds=probs.argmax(1)
   for j in range(len(y)):
    truth=y[j].item(); pred=preds[j].item(); score=probs[j,pred].item(); path=b.samples[idx][0];idx+=1
    rows.append([str(Path(path).relative_to(ROOT)),a.classes[truth],a.classes[pred],f'{score:.4f}',int(truth==pred)])
    cm[truth][pred]+=1;total+=1;correct+=int(truth==pred)
 out=ROOT/'results';out.mkdir(exist_ok=True)
 with (out/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['file','actual','predicted','confidence','correct']);w.writerows(rows)
 with (out/'confusion_matrix.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['actual\\predicted']+a.classes)
  for name,arr in zip(a.classes,cm):w.writerow([name]+arr)
 report={'test_images':total,'correct':correct,'accuracy':round(correct/total,4),'accuracy_percent':round(100*correct/total,1),'per_class_accuracy':{a.classes[i]:round(cm[i][i]/sum(cm[i]),4) for i in range(len(cm))},'warning':'This is test-set accuracy on the seven included categories, NOT accuracy on all foods.'}
 (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print('Test:',report)
 modeldir=ROOT/'model';modeldir.mkdir(exist_ok=True)
 torch.save({'model':net.cpu().state_dict(),'classes':a.classes},modeldir/'japanese_food_model.pt')
 net.eval()
 torch.onnx.export(net,torch.randn(1,3,224,224),str(modeldir/'japanese_food_model.onnx'),input_names=['images'],output_names=['logits'],opset_version=17)
 (modeldir/'labels.json').write_text(json.dumps({'classes':a.classes,'japanese':JA,'image_size':224,'resize_short_side':256,'mean':MEAN,'std':STD},ensure_ascii=False,indent=2),encoding='utf-8')
 print('Saved ONNX, labels, CSV, JSON. Open web/index.html via Live Server after copying model/*.onnx and labels.json to web/model/.')

if __name__=='__main__':main()
