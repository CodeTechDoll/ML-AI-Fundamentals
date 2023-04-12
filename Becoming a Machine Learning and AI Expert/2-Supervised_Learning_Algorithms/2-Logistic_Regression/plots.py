import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(df.iloc[:, :4])

# Fit logistic regression model
model = LogisticRegression()
model.fit(X, iris.target)

# Function to plot decision boundary
def plot_decision_boundary(X, y, model, feature1, feature2):
    x_min, x_max = X[:, feature1].min() - 1, X[:, feature1].max() + 1
    y_min, y_max = X[:, feature2].min() - 1, X[:, feature2].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = go.Figure()

    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='Viridis',
                            showscale=False, opacity=0.4, name='Decision Boundary'))

    for species, color in zip(range(3), px.colors.qualitative.Plotly):
        fig.add_trace(go.Scatter(x=X[y == species, feature1], y=X[y == species, feature2],
                                mode='markers', marker=dict(color=color, size=8),
                                name=iris.target_names[species]))

    fig.update_layout(scene=dict(xaxis_title=f'Feature {feature1}', yaxis_title=f'Feature {feature2}'))

    return fig

