f=@ (x) x(1)^2+x(1)^4+x(2)^6-x(2)^2+2*(x(1)*x(2));
F=@ (x,y) f([x,y]);
figure;fmesh(F,[-2,2])
figure; fcontour(F,[-2,2])
axis equal
hold on;
% Define the domain % Create a grid of x and y values % Compute the gradient of the function
x1 = linspace(-2,2,25);
x2=x1;
[X1, X2] = meshgrid(x1,x2);
F1 = 2*X1 + 4*X1.^3 + 2*X2;
F2 = 6*X2.^5 - 2*X2 + 2*X1;
quiver(X1, X2, F1, F2)
%gradient function
g=@(x) [2*x(1)+4*x(1)^3+2*x(2); 6*x(2)^5-2*x(2)+2*x(1)];
%the hessian 
h=@(x) [2+12*x(1)^2,2 ; 2,30*x(2)^4-2];
% The three methods grdient, Newton, and BFGS
[xoptG,foptG,kG]=mygradient([-1.0;1.0],1e-3,3000,f,g)
[xoptN,foptN,k2]= mynew([-1.0;1.0], f,g,h,1e-3,3000)
[xoptBFGS,foptBFGS,kBFGS]=mybfgs([-1.0;1.0],f,g)

function [xoptB,foptB,kB]=mybfgs(x0,f,g,e,maxit)
    if nargin<4
        e=1e-3;
    end
    if nargin<5
        maxit=200;
    end
    if ~iscolumn(x0)
        error('the initial vector has to be a column vector')
    end
    kB=0;
    H=eye(length(x0));
    H0=H;
    while norm(g(x0))>e
        p=-H*g(x0);
        alpha=backtBFGS(0.5,0.01,0.5,f,g,p,x0);
        x1=x0+alpha*p;
        kB=kB+1;
        if kB>maxit
            error('maxit reached')
        end
        s=x1-x0;
        y=g(x1)-g(x0);
        H=(H0-s*y'/dot(y,s))*H*(H0-y*s'/dot(y,s))+s*s'/dot(y,s);
        plot([x0(1),x1(1)],[x0(2),x1(2)],'k*-');
        x0=x1;
    end
    xoptB=x1;
    foptB=f(xoptB);
end


function alpha=backtBFGS(alpha,c,rho,f,g,p,x)
  while f(x+alpha*p)>f(x)+alpha*c*dot(g(x),p)
    alpha=rho*alpha;
  end
end

%  the inputs xopt, fopt, k = the name of the function
function [xoptN, foptN,kN] = mynew(x0,f,g,h,e,maxit)
   kN=0;
    while norm(g(x0))>e
       p= -h(x0)\g(x0);
       x1=x0 +p
       kN=kN+1
       if kN >maxit
           error('increase the value of maxit')
       end
       plot([x0(1),x1(1)],[x0(2),x1(2)],'g*-')
       x0= x1
    end
    xoptN=x1;
    foptN= f(xoptN); % the optimal value for the function

end

function [xoptG,foptG,kG]=mygradient(x0,e,maxit,f,g)
  kG=0;
  if ~iscolumn(x0)
        error('The initial vector has to be a column vector')
  end
  while norm(g(x0))>e
     alpha=backtG(0.4,0.01,0.4,f,g,x0);
     x=x0-alpha*g(x0);
     plot([x(1),x0(1)],[x(2),x0(2)],'b*-')
     x0=x
     kG=kG+1
     if kG>maxit
        error('The number of steps exceeds the maxit')
     end
  end
  xoptG=x;
  foptG=f(xoptG);
end

function alpha=backtG(alpha,c,rho,f,g,x)
    while f(x-alpha*g(x))>f(x)-alpha*c*dot(g(x),g(x))
        alpha=rho*alpha;
    end
end
