// Generates the unrolled Hex8 kernels (KE for h=1, nu=0.3) spliced into lite/solver.js.
// Usage: node lite/tools/gen_kernel.js   (rewrites the block between the GENERATED KERNELS markers)
const fs=require("fs");
const path=require("path").join(__dirname,"..","solver.js");
const src=fs.readFileSync(path,"utf8");
global.self={};eval(src.replace(/^function topopWorkerMain\(\) \{/m,"function topopWorkerMain(){ globalThis.hexKE=hexKE;"));topopWorkerMain();
const K=hexKE(1,0.3);
const vals=[];
const ref=(v)=>{const a=Math.abs(v);let f=vals.findIndex(x=>Math.abs(x-a)<1e-12*1.25);if(f<0){vals.push(a);f=vals.length-1;}return (v<0?"-":"+")+"k"+f;};
const row=(i)=>{let t="";for(let j=0;j<24;j++)t+=ref(K[i*24+j])+"*u"+j;return t.replace(/^\+/,"");};
const rows=[];for(let i=0;i<24;i++)rows.push(row(i));
const consts=vals.map((v,i)=>`k${i} = ${v.toPrecision(17)}`).join(", ");
let loadU="";for(let a=0;a<8;a++)loadU+=`      const a${a} = 3 * en[b + ${a}];\n      const u${3*a} = p[a${a}], u${3*a+1} = p[a${a} + 1], u${3*a+2} = p[a${a} + 2];\n`;
let mv=`  // BEGIN GENERATED KERNELS: KE(h = 1, nu = 0.3) unrolled; selfTest() checks them against hexKE().\n`;
mv+=`  const ${consts};\n`;
mv+=`  function mvKernel(p, out, en, Eh, live, nl) {\n    for (let q = 0; q < nl; q++) {\n      const k = live[q];\n      const b = 8 * k;\n      const E = Eh[k];\n${loadU}`;
for(let i=0;i<24;i++){const a=Math.floor(i/3),d=i%3;mv+=`      out[a${a}${d?" + "+d:""}] += E * (${rows[i]});\n`;}
mv+=`    }\n  }\n`;
mv+=`  function ceKernel(p, en, ne, ce) {\n    for (let k = 0; k < ne; k++) {\n      const b = 8 * k;\n${loadU}      ce[k] =\n`;
mv+=rows.map((r,i)=>`        u${i} * (${r})`).join(" +\n")+";\n    }\n  }\n  // END GENERATED KERNELS\n";
const out=src.replace(/  \/\/ BEGIN GENERATED KERNELS[\s\S]*?\/\/ END GENERATED KERNELS\n/, mv);
if(out===src && !src.includes("BEGIN GENERATED")) throw new Error("marker missing");
fs.writeFileSync(path,out);
console.log("vals",vals.length,"bytes",mv.length);
