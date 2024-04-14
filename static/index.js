let bar=document.getElementById("bars");
let div=document.getElementById("menu");
let x=true;
bar.addEventListener('click',()=>{
    if(x){
        div.style.display="none";// adding toggle mode
    }
    else{
        div.style.display="block";
    }
    x=!x;
});
let select=document.getElementById('lang');
for(let y in humanLanguages){
    let option=document.createElement('option');
    option.innerText=humanLanguages[y];
    select.append(option);
}
let globe=document.getElementById('globe');
let a=true;
globe.addEventListener('click',()=>{
if(a){
    select.style.display='none';
}
else{
    select.style.display='block';
}
a=!a;
});
let pro=document.getElementById('pro');
let click=document.getElementById('click');
let b=true;
click.addEventListener('click',()=>{
    if(b){
        pro.style.display='none'
    }
    else{
        pro.style.display='block'
    }
    b=!b;
})
let p=document.getElementById('ada');
let das=document.getElementById('das');
let dark='light';
das.addEventListener('click',()=>{
    if(dark){
        pro.style.backgroundColor='white';
        pro.style.color='black';
        p.innerText='Light mode';
        das.innerText='Light mode'
    }
    else{
        
        pro.style.backgroundColor='black';
        pro.style.color='white';
        p.innerText='Dark mode';
        das.innerText='Dark mode';
    }
    dark=!dark;
})

let sal=document.getElementById('perc');
console.log(sal);
for(let i=0;i<items.length;i++){
    
}
