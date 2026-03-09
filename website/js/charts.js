setTimeout(()=>{

let data = window.resultData

let classLabels = Object.keys(data.class_distribution)
let classValues = Object.values(data.class_distribution)

new Chart(document.getElementById("classChart"),{
type:"bar",
data:{
labels:classLabels,
datasets:[{
label:"Objects",
data:classValues
}]
}
})

let featLabels = Object.keys(data.feature_importance)
let featValues = Object.values(data.feature_importance)

new Chart(document.getElementById("featureChart"),{
type:"bar",
data:{
labels:featLabels,
datasets:[{
label:"Importance",
data:featValues
}]
}
})

},1000)