#https://scikitlearnjl.readthedocs.io/en/latest/quickstart/

using ScikitLearn
using RDatasets: dataset

iris = dataset("datasets", "iris")

# ScikitLearn.jl expects arrays, but DataFrames can also be used - see
# the corresponding section of the manual
X = convert(Array, iris[[:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = convert(Array, iris[:Species])

@sk_import linear_model: LogisticRegression
model = LogisticRegression(fit_intercept=true)
fit!(model, X, y)

accuracy = sum(predict(model, X) .== y) / length(y)
println("accuracy: $accuracy")

using ScikitLearn.CrossValidation: cross_val_score

# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_validation.py
cross_val_score(LogisticRegression(), X, y; cv=5)  # 5-fold
