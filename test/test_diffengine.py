import torch
from scipy.special import gamma
from omnidiff.diffengine import Value
def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.tanh() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).tanh()
    d += 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).tanh()
    d = d + 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
def test_custom_ops():
    a = Value(4.0)
    b=a.ln()
    k=math.log(4.0)
    c=b.abs()
    s= a.softsign()
    s.backward()
    o=Value(1.5)
    gam=o.gamma_actual()
    gam.backward()
    p= Value(4.0)
    rel=p.reLU()
    rel.backward()
    tol=1e-6

    #pytorch for softsign
    a1=torch.Tensor([4.0]).double()
    a1.requires_grad=True
    m=torch.nn.Softsign()
    s1=m(a1)
    s1.backward()
    p1=torch.Tensor([4.0]).double()
    p1.requires_grad=True
    k1=p1.relu()
    k1.backward()

    #scipy for gamma
    gam1= gamma(1.5)

    assert abs(b.data-k) <tol
    assert abs(c.data-abs(b.data))<tol
    assert abs(s.data-s1.data.item())<tol
    assert abs(a.grad-a1.grad.item())<tol
    assert abs(gam.data-gam1)<tol
    assert abs(rel.data-k1.data.item())<tol
    assert abs(p1.grad - p1.grad.item())< tol