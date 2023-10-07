
# Documentation of Multiple linear regression model (without bias*)

Let's get the theory super right first.

## Features

Our input are multiple features:
$x_1, x_2, x_3, \ldots, x_n $

We can write that in terms of a vector as:

$$ \vec{X} = \begin{pmatrix}
    x_1 \\ 
    x_2 \\ 
    x_3 \\ 
    \vdots \\ 
    x_n 
\end{pmatrix} $$

## Parameters

Our model parameters the set of weights for each feature. (We don't have bias, but there is a workaround*😉) Listing them: 
$w_1, w_2, w_3, ..., w_n $

We can also write them in the form of vectors as:

$$ \vec{W} = \begin{pmatrix} 
    w_1 \\ 
    w_2 \\ 
    w_3 \\ 
    \vdots 
    \\ w_n 
\end{pmatrix} $$

## Model

Thus, from our given inputs and the parameters, our model functon is:

${ f( \vec{X} ) = w_1 x_1 + w_2 x_2 + w_3 x_3 + \cdots + w_n x_n}$

Thus we can simplify the model as:

$$ f(\vec{X}) = \begin{pmatrix} 
    w_1 \\ 
    w_2 \\ 
    \vdots \\ 
    w_n 
\end{pmatrix} \cdot \begin{pmatrix} 
    x_1 \\ 
    x_2 \\ 
    \vdots \\ 
    x_n
\end{pmatrix} $$

$$ f(\vec X) = \vec W \cdot \vec X$$


## Loss function

One of the simplest loss function we can use is the Mean Squared Error (MSE)

For a single example let's say $(\vec{X}, y)$:

Loss function is 
${J(\vec{W}) = \displaystyle \frac{(f(\vec{X}) - y)^2}{2}}$

For $m$ training examples: 
$ {(\vec{X_1}, y_1), (\vec{X_2}, y_2), (\vec{X_3}, y_3), \ldots,  (\vec{X_m}, y_m) }$

The loss function is:

$J(\vec{W}) = \displaystyle \frac{
    \left(f(\vec{X_1}) - y_1\right)^2 + 
    \left(f(\vec{X_2}) - y_2\right)^2 + 
    \cdots + 
    \left(f(\vec{X_m}) - y_m\right)^2
}{2m}$

Which is literally the half of the mean of squares of the errors.
The errors are also called residuals.

Residual for $i^{th}$ example = Observed Value $-$ Predicted Value

$e_i = y_i - f(\vec{X}_i)$

Note that squaring the result makes the preceeding subtraction commutative (i.e. $(a-b)^2$ = $(b-a)^2$).

Now, the loss function can be simplified as a simple function of summation as:

$$J(\vec{W}) = \displaystyle \frac1{2m}\sum_{i=1}^{m}\left( f(\vec{X_i}) - y_i  \right)^2 $$


## Gradient of the loss function

We can calculate it as:

$\nabla J(\vec W) = \begin{pmatrix} 
    \displaystyle \frac {\partial J(\vec W)}{\partial w_1} \\
    \displaystyle \frac{\partial J(\vec W)}{\partial w_2} \\
    \vdots \\ 
    \displaystyle \frac {\partial J(\vec W)}{\partial w_n} \\
\end{pmatrix}$

$\begin{aligned}
    \displaystyle \frac {\partial J(\vec W)} {\partial w_1} & =  \displaystyle \frac{\partial}{\partial w_1} \left(\frac1{2m}\sum_{i=1}^{m}\left( f(\vec{X_i}) - y_i  \right)^2 \right) \\
    & = \displaystyle \frac1{2m}\sum_{i=1}^{m}\frac{\partial}{\partial w_1}  \left( f(\vec{X_i}) - y_i  \right)^2  \\
    & = \displaystyle \frac1{2m}\sum_{i=1}^{m} 2 \left( f(\vec{X_i}) - y_i  \right)^1 \frac{\partial}{\partial w_1}  \left( f(\vec{X_i}) - y_i  \right)  \\
    & = \displaystyle \frac1{m}\sum_{i=1}^{m} \left( f(\vec{X_i}) - y_i  \right) \frac{\partial f(\vec{X_i})}{\partial w_1}\\
    & = \displaystyle \frac1{m}\sum_{i=1}^{m} \left( f(\vec{X_i}) - y_i  \right) \frac{\partial \left( \vec W \cdot \vec X_i \right)}{\partial w_1}\\
    & = \displaystyle \frac1{m}\sum_{i=1}^{m} \left( f(\vec{X_i}) - y_i  \right) \frac{\partial \left( w_1 x_1^{(i)} + w_2 x_2^{(i)} + \cdots + w_n x_n^{(i)}\right)}{\partial w_1}\\
    & = \displaystyle \frac1{m}\sum_{i=1}^{m} \left( f(\vec{X_i}) - y_i  \right) x_1^{(i)}\\
\end{aligned}$

Thus,

$\begin{aligned}
\nabla J(\vec W) & = \begin{pmatrix} 
    \displaystyle \frac1{m}\sum_{i=1}^{m} \left( f(\vec{X_i}) - y_i  \right) x_1^{(i)} \\
    \displaystyle \frac1{m}\sum_{i=1}^{m} \left( f(\vec{X_i}) - y_i  \right) x_2^{(i)} \\
    \vdots \\
    \displaystyle \frac1{m}\sum_{i=1}^{m} \left( f(\vec{X_i}) - y_i  \right) x_n^{(i)}
\end{pmatrix} \\
& = \displaystyle \frac1m \begin{pmatrix} 
    \displaystyle \sum_{i=1}^{m} \left( f(\vec{X_i}) - y_i  \right) x_1^{(i)} \\
    \displaystyle \sum_{i=1}^{m} \left( f(\vec{X_i}) - y_i  \right) x_2^{(i)} \\
    \vdots \\
    \displaystyle \sum_{i=1}^{m} \left( f(\vec{X_i}) - y_i  \right) x_n^{(i)}
\end{pmatrix} \\
& = \displaystyle \frac1m \sum_{i=1}^{m} \begin{pmatrix} 
    \displaystyle \left( f(\vec{X_i}) - y_i  \right) x_1^{(i)} \\
    \displaystyle \left( f(\vec{X_i}) - y_i  \right) x_2^{(i)} \\
    \vdots \\
    \displaystyle \left( f(\vec{X_i}) - y_i  \right) x_n^{(i)}
\end{pmatrix} \\
& = \displaystyle \frac1{m}\sum_{i=1}^{m} \left( f(\vec{X_i}) - y_i  \right) \begin{pmatrix}
    x_1^{(i)} \\
    x_2^{(i)} \\
    \vdots \\
    x_n^{(i)} \\
\end{pmatrix}\\
& = \displaystyle \frac1{m}\sum_{i=1}^{m}  \left( f(\vec{X_i}) - y_i  \right) \overrightarrow X^{(i)}
 \\
\end{aligned}$

Thus the final expression is:
$$\nabla J(\vec W) = \displaystyle \frac1{m}\sum_{i=1}^{m}  \left( f(\vec{X_i}) - y_i  \right) \overrightarrow X^{(i)}
$$


## *Adding Bias
If we add an extra feature $x_0$, which has a constant value (let's say $1$), then ***the weight*** corresponding to the feature, ${w_0}$ becomes the bias.

Thus, for our model,

$f(\vec X) = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n $ ( since $x_0 = 1, w_0 x_0 = w_0$)

By doing this, none of the formulas are affected. But we incur a little cost of addition of single extra feature for all the examples.

That was one of the way. Another way is to add bias manually.
We add bias to the existing model as:

$f_{new}(\vec X) = f(\vec X) + b $

Now, to calculate the gradient, for all the weights, the gradient calculation step remains the same. For the bias, we calculate the gradient separately as:
$ \displaystyle \frac{\partial J(\overrightarrow W, b)}{\partial b} = \sum_{i=1}^{m}\left(f(\overrightarrow X_i) - y_i\right) $

## Gradient Descent

It's very simple.

We have a learning rate = $\alpha$, and the usual gradient of the loss function $\nabla J (\vec W) $.

Using those values, we get the gradient descent algorithm as the simple iteration of the following step.

$$\vec W_{new} = \vec W - \alpha \nabla J (\vec W)$$
$$W = W_{new}$$

## Regularization of the parameters

They can be done pretty easily by adding a regularization parameters.

### L1 Regularization

$$J(\vec{W}) = \displaystyle \frac1{2m}\sum_{i=1}^{m}\left( f(\vec{X_i}) - y_i  \right)^2  + \frac\lambda{m} \lVert \vec W \rVert_1$$
where, $ \displaystyle \lVert \vec W \rVert_1 = \sum_{j=1}^{n}\vert w_j \vert $ is the L1 norm of the weight vector $\vec W$.

$$\nabla J(\vec W) = \displaystyle \frac1{m}\sum_{i=1}^{m}  \left( f(\vec{X_i}) - y_i  \right) \overrightarrow X^{(i)} + \frac\lambda{m}\begin{pmatrix} \displaystyle \frac{\partial |w_1|}{\partial w_1} \\ \displaystyle \frac{\partial |w_2|}{\partial w_2} \\ \vdots \\ \displaystyle \frac{\partial |w_n|}{\partial w_n} \end{pmatrix}
$$

### L2 Regularization

$$J(\vec{W}) = \displaystyle \frac1{2m}\sum_{i=1}^{m}\left( f(\vec{X_i}) - y_i  \right)^2  + \frac\lambda{2m} \lVert \vec W \rVert_2$$
where, $ \displaystyle \lVert \vec W \rVert_2 = \sum_{j=1}^{n} w_j^2 $ is the L2 norm of the weight vector $\vec W$.

Now, taking gradient of the regularization term, we get:

$ \begin{aligned}
    \displaystyle \nabla \frac{\lambda}{2m} \Vert \vec W \Vert _2 
    & = \frac{\lambda}{2m}\nabla \left( w_1^2 + w_2^2 + \cdots + w_n^2 \right) \\
    & = \frac{\lambda}{2m} \begin{pmatrix}
            2 w_1 \\ 2 w_2 \\ \vdots \\ 2 w_n
        \end{pmatrix} \\
    & = \frac{\lambda}{m} \begin{pmatrix}
            w_1 \\ w_2 \\ \vdots \\ w_n 
        \end{pmatrix} \\
    & = \frac{\lambda}{m} \overrightarrow W \\
\end{aligned}$

$$\nabla J(\vec W) = \displaystyle \frac1{m}\sum_{i=1}^{m}  \left( f(\vec{X_i}) - y_i  \right) \overrightarrow X^{(i)} + \frac{\lambda}{m} \overrightarrow W
$$
