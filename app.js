const $ = id => document.getElementById(id);
let imageUrl;
$('photo').addEventListener('change', () => {
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  const file = $('photo').files[0];
  if (!file) return;
  imageUrl = URL.createObjectURL(file); $('preview').src = imageUrl;
  $('result').textContent = '';
});
const softmax = a => {const m=Math.max(...a), e=a.map(x=>Math.exp(x-m));const s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s)};
async function loadImage(img) {if (!img.complete || !img.naturalWidth) await new Promise((res,rej)=>{img.onload=res;img.onerror=rej})}
function tensorFromImage(img) {
  // torchvision Resize(256): shorter side=256, followed by CenterCrop(224).
  const w=img.naturalWidth,h=img.naturalHeight, scale=256/Math.min(w,h);
  const nw=Math.round(w*scale), nh=Math.round(h*scale);
  const tmp=document.createElement('canvas');tmp.width=nw;tmp.height=nh;
  tmp.getContext('2d').drawImage(img,0,0,nw,nh);
  const crop=document.createElement('canvas');crop.width=224;crop.height=224;
  const ctx=crop.getContext('2d',{willReadFrequently:true});
  ctx.drawImage(tmp,Math.round((nw-224)/2),Math.round((nh-224)/2),224,224,0,0,224,224);
  const px=ctx.getImageData(0,0,224,224).data, a=new Float32Array(3*224*224), mean=[.485,.456,.406],std=[.229,.224,.225];
  for(let i=0;i<224*224;i++) for(let c=0;c<3;c++)a[c*224*224+i]=((px[4*i+c]/255)-mean[c])/std[c];
  return new ort.Tensor('float32',a,[1,3,224,224]);
}
$('predict').addEventListener('click',async()=>{
  if(!$('photo').files[0]) {$('status').textContent='先に写真を選んでね。';return}
  $('predict').disabled=true; $('status').textContent='AIモデルを読み込み中…';
  try {
    if(!window.ort) throw new Error('ONNX Runtimeを読み込めません。ネット接続を確認してね。');
    const resp=await fetch('model/labels.json');if(!resp.ok)throw new Error('web/model/labels.json がありません。');
    const info=await resp.json();
    await loadImage($('preview'));
    const session=await ort.InferenceSession.create('model/japanese_food_model.onnx',{executionProviders:['wasm']});
    const prediction=await session.run({images:tensorFromImage($('preview'))});
    const scores=softmax(Array.from(prediction.logits.data));
    const ranking=scores.map((s,i)=>({name:info.japanese[info.classes[i]]||info.classes[i],score:s})).sort((a,b)=>b.score-a.score);
    const [first,second]=ranking;
    const uncertain=(first.score<.65 || first.score-second.score<.20);
    $('result').textContent=(uncertain?'⚠️ 判定が不確かです。手動で確認してください。\n':'')+ranking.slice(0,3).map((x,i)=>`${i+1}位 ${x.name}：${(x.score*100).toFixed(1)}%`).join('\n');
    $('status').textContent='判定完了（対応7分類・実験用）';
  } catch(e) {$('status').textContent='読み込み失敗: '+e.message;}
  finally {$('predict').disabled=false;}
});
