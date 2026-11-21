import fs from"node:fs";
const I=process.env.MAKEFUN_INPUT_DIR||"/input",O=process.env.MAKEFUN_OUTPUT_DIR||"/output";
let w,N,V,W,D,G;
function bfs(s,o){
let p=new Int32Array(N).fill(-1),q=[s];p[s]=s;
for(let i=0;i<q.length;i++)for(let d of D){let y=q[i]+d;
if(!W[y]&&!(o&&o[y])&&p[y]<0){p[y]=q[i];q.push(y);}}
return p;
}
function solve(b0,p0){
let S=[[b0,p0,""]],seen=new Set([b0+":"+p0]),qi=0;
while(qi<S.length){
let s=S[qi++],b=s[0],p=s[1];
if(b.every(x=>G.includes(x)))return{m:s[2],e:p};
let o=new Uint8Array(N);for(let x of b)o[x]=1;
let r=bfs(p,o);
for(let bi=0;bi<b.length;bi++){let x=b[bi];
for(let d of D){let t=x+d;
if(r[x-d]<0||W[t]||o[t])continue;
let ws="",y=x-d;
for(;y!=p;y=r[y])ws="rldu"[D.indexOf(y-r[y])]+ws;
let nb=b.slice();nb[bi]=t;nb.sort((a,c)=>a-c);
if(seen.has(nb+":"+x))continue;
seen.add(nb+":"+x);
S.push([nb,x,s[2]+ws+"rldu"[D.indexOf(d)]]);
}}
}
return null;
}
function parse(t){
w=t.indexOf("\n");let s=t.replace(/\n/g,"");N=s.length;
V=new Uint8Array(N).fill(1);D=[1,-1,w,-w];
let B=[],Gl=[],P=-1;
for(let i=0;i<N;i++){let c=s[i];
if(c!="#"){V[i]=0;if(".*+".includes(c))Gl.push(i);if("$*".includes(c))B.push(i);if("@+".includes(c))P=i;}}
B.sort((x,y)=>x-y);
return{B,G:Gl,P};
}
function rooms(L){
let r=(L.P/w)|0;W=V.slice().fill(1,r*w,(r+1)*w);
let v=new Uint8Array(N),R=[];
for(let s=0;s<N;s++){
if(W[s]||v[s])continue;
let p=bfs(s,0),b=[],g=[];
for(let i=0;i<N;i++)if(p[i]>=0)v[i]=1;
for(let x of L.B)if(p[x]>=0)b.push(x);
for(let x of L.G)if(p[x]>=0)g.push(x);
R.push([b,g,0]);
}
if(R.some(x=>x[0].length!=x[1].length))return null;
return R.filter(x=>x[0].length);
}
function solveLevel(L){
let R=rooms(L);
if(!R){
if(L.B.length>5)return"";
W=V.slice();G=L.G;
let s=solve(L.B,L.P);return s?s.m:"";
}
let m="",pl=L.P;
for(let r of R){
W=V.slice();
for(let o of R)if(o!==r)for(let b of o[o[2]?1:0])W[b]=1;
G=r[1];
let s=solve(r[0],pl);
if(!s)return m;
m+=s.m;pl=s.e;r[2]=1;
}
return m;
}
for(let i of fs.readdirSync(I)){
let p=I+"/"+i,m="";
try{m=solveLevel(parse(fs.readFileSync(p+"/level.txt","utf8")))}catch(e){}
p=O+"/"+i;fs.mkdirSync(p,{recursive:true});fs.writeFileSync(p+"/solution.txt",m+"\n");
}
