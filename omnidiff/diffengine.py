class Value:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data=data
        self._prev=set(_children)
        self._op=_op
        self.label=label
        self.grad=0.0
        self._backward= lambda:None
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad+=1.0*out.grad
            other.grad+=1.0*out.grad
        out._backward=_backward
        return out
    def __radd__(self,other):
        return self+other
    def __neg__(self): 
        return self * -1
    def __sub__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out=Value(self.data-other.data,(self,other),'-')
        def _backward():
            self.grad+=1.0*out.grad
            other.grad+=-1.0*out.grad
        out._backward=_backward
        return out
    def __rsub__(self,other):
        return -1*(self-other)
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out
    def __rmul__(self,other):
        return self*other
    def __truediv__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out=Value(self.data/other.data,(self,other),'/')
        def _backward():
            self.grad+=(1/other.data)*out.grad
            other.grad+=((-1*self.data)/(other.data**2))*out.grad
        out._backward=_backward
        return out
    def __rtruediv__(self,other):
        return Value(1)/(self/other)
    def __pow__(self,other):
        assert isinstance(other,(int,float)), "only supporting int/float for now"
        out=Value(self.data**other,(self,),'power')
        def _backward():
            self.grad += (other*(self.data**(other-1)))*out.grad
        out._backward=_backward
        return out
    def tanh(self):
        x=self.data
        exp= (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out=Value(exp,(self,),'tanh')
        def _backward():
            self.grad+=(1-(exp**2))*out.grad
        out._backward=_backward
        return out
    def exp(self):
        x=self.data
        exp=math.exp(x)
        out= Value(exp,(self,),'exp')
        def _backward():
            self.grad+=exp*out.grad
        out._backward=_backward
        return out
    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for k in v._prev:
                    build_topo(k)
                topo.append(v)
        self.grad=1.0
        build_topo(self)
        for i in reversed(topo):
            i._backward()
            
            
    def ln(self):
        x=self.data
        out=Value(math.log(x),(self,),'ln()')
        def _backward():
            self.grad+=(1/x)*out.grad
        out._backward=_backward
        return out
    def sin(self):
        x=self.data
        out=Value(math.sin(x),(self,),'sin()')
        def _backward():
            self.grad+=(math.cos(x))*out.grad
        out._backward=_backward
        return out
    def cos(self):
        x=self.data
        out=Value(math.cos(x),(self,),'cos()')
        def _backward():
            self.grad+=(-1*math.sin(x))*out.grad
        out._backward=_backward
        return out
    def sqrt(self):
        x=self.data
        out=Value(math.sqrt(x),(self,),'sqrt')
        def _backward():
            self.grad+=(1/(2*math.sqrt(x)))*out.grad
        out._backward=_backward
        return out
    def tan(self):
        x=self.data
        out=Value(math.tan(x),(self,),'tan')
        def _backward():
            self.grad+=(1/(math.cos(x)**2))*out.grad
        out._backward=_backward
        return out
    def abs(self):
        x=self.data
        out=Value(abs(x),(self,),'abs')
        def _backward():
            if x>0:
                self.grad+=(1)*out.grad
            elif x<0:
                self.grad+=-1*out.grad
            else:
                self.grad+=0
        out._backward=_backward
        return out
    def reLU(self):
        x=self.data
        out=Value(max(x,0),(self,),'relu')
        def _backward():
            if x>0:
                self.grad+=out.grad
            else:
                self.grad+=0
        out._backward=_backward
        return out
    def sigmod(self):
        x=self.data
        sigmod=(1/(1+math.exp(-x)))
        out=Value(sigmod,(self,),'sigmod')
        def _backward():
            self.grad+=(sigmod*(1-sigmod))*out.grad
        out._backward=_backward
        return out
    def gamma_internal(self,z,upperBound=100,n=100000):
        if(z<=0):
            raise ValueError("Node data must be positive")
        a=0.000001
        b=upperBound
        h= (b-a)/n

        def integrand(t):
            return (t**(z-1))*math.exp(-t)
        sum= 0.5*(integrand(a)+integrand(b))
        for i in range(1,n):
            t= a+i*h
            sum+=integrand(t)
        sum=sum*h
        return sum
    def digamma_internal(self,x):
        h=0.000001
        k=(math.log(self.gamma_internal(x+h))-math.log(self.gamma_internal(x-h)))/h
        return k
    def gamma_actual(self,z_a=None, upperBound_a=100, n_a=1000000):
        u1=upperBound_a
        n1=n_a
        if z_a==None:
            z_a=self.data
        if z_a>0:
            k=self.gamma_internal(z_a,u1,n1)
            out=Value(k,(self,),'gamma')
            def _backward():
                self.grad+=(k*self.digamma_internal(z_a))*out.grad
            out._backward=_backward
            return out
        elif z_a==int(z_a):
            raise ValueError('Gamma is not for negative integers')
        else:
            return self.gamma_actual(z_a=z_a+1)/z_a
    def softsign(self):
        x=self.data
        output=x/(1+abs(x))
        out= Value(output,(self,),'softsign')
        def _backward():
            if x>0:
                self.grad+=(1/((1+x)**2))*out.grad
            elif x<0:
                self.grad+=(1/((1-x)**2))*out.grad
            else:
                self.grad+=0
        out._backward=_backward
        return out
            
    