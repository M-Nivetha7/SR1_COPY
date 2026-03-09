async function loadData(){

let res = await fetch("../results/results.json")
let data = await res.json()

document.getElementById("rows").innerText = data.rows
document.getElementById("accuracy").innerText = (data.accuracy*100).toFixed(2)+"%"

let headers = Object.keys(data.preview[0])

let thead = document.querySelector("#table thead")
let tbody = document.querySelector("#table tbody")

let headrow = "<tr>"

headers.forEach(h=>{
headrow += `<th>${h}</th>`
})

headrow += "</tr>"

thead.innerHTML = headrow

data.preview.forEach(r=>{

let row="<tr>"

headers.forEach(h=>{
row+=`<td>${r[h]}</td>`
})

row+="</tr>"

tbody.innerHTML+=row

})

window.resultData = data

}

loadData()