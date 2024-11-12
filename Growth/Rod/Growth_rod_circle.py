"""
@Problem: 2D Cook's membrane. A benchmark example.

@Formulation: Mixed displacement-pressure formulation.

@Material: Incompressible Linear elastic.

@author: Dr Chennakesava Kadapa

Created on Sun 18-Jun-2024
"""


# Import FEnicSx/dolfinx
import dolfinx

# For numerical arrays
import numpy as np

# For MPI-based parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# PETSc solvers
from petsc4py import PETSc

# specific functions from dolfinx modules
from dolfinx import fem, mesh, io, plot, log, default_scalar_type
from dolfinx.fem import (Constant, dirichletbc, Function, functionspace, Expression )
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile

# specific functions from ufl modules
import ufl
from ufl import (TestFunctions, TrialFunction, Identity, grad, sym, det, div, dev, inv, tr, sqrt, conditional ,\
                 gt, dx, ds, inner, derivative, dot, ln, split, exp, eq, cos, acos, ge, le)

# basix finite elements
import basix
from basix.ufl import element, mixed_element, quadrature_element




msh, markers, facet_tags = io.gmshio.read_from_msh("rod3d-H27-nelem50.msh", MPI.COMM_WORLD)

# coordinates of the nodes
x = ufl.SpatialCoordinate(msh)


# Define the volume integration measure "dx" 
# also specify the number of volume quadrature points.
quadrature_degree = 4
dx = ufl.Measure('dx', domain=msh, metadata={'quadrature_degree': quadrature_degree})
ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tags, metadata={'quadrature_degree': quadrature_degree})



# FE Elements
# Quadratic element for displacement
###
###
# Define function spaces and elements
deg_u = 2
deg_p = deg_u-1

# displacement
U2 = element("Lagrange", msh.basix_cell(), deg_u, shape=(3,))
# pressure
P1 = element("Lagrange", msh.basix_cell(), deg_p)

# Mixed element
TH = mixed_element([U2, P1])
ME = functionspace(msh, TH)


V2 = functionspace(msh, U2) # Vector function space


# functions with DOFs at the current step
w = Function(ME)
u, p = split(w)

# functions with DOFs at the previous step
w_old = Function(ME)
u_old, p_old = split(w_old)

# functions with DOFs at the previous step
w_old2 = Function(ME)
u_old2, p_old2 = split(w_old2)

# Test functions
u_test, p_test = TestFunctions(ME)

# Trial functions
dw = TrialFunction(ME)


# Boundary conditions
dofs_1_x = fem.locate_dofs_topological(ME.sub(0).sub(0), facet_tags.dim, facet_tags.find(1)) #ux
dofs_1_y = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(1)) #uy
dofs_1_z = fem.locate_dofs_topological(ME.sub(0).sub(2), facet_tags.dim, facet_tags.find(1)) #uz

bc1_x = fem.dirichletbc(value=0.0, dofs=dofs_1_x, V=ME.sub(0).sub(0))
bc1_y = fem.dirichletbc(value=0.0, dofs=dofs_1_y, V=ME.sub(0).sub(1))
bc1_z = fem.dirichletbc(value=0.0, dofs=dofs_1_z, V=ME.sub(0).sub(2))

bcs = [bc1_x, bc1_y, bc1_z]


# Parameter values
E = 1e6  # MPa
nu = 0.4999 # Poisson's ratio
G = E/(2*(1 + nu))

eps = (3*(1 - 2*nu))/E 

mu = Constant(msh, G)
kappa_inv = Constant(msh, PETSc.ScalarType(eps))


t = 0.0
dt = 0.02
time_final = 1.0
time_cur = 0.0



# dimension
d = len(u)

# Kinematics
Id = ufl.variable(Identity(3))

# Total deformation gradient
F  = ufl.variable(Id + grad(u))

C  = ufl.variable(F.T*F)
J  = ufl.variable(det(F))

Fbar = ufl.variable(J**(-1.0/3)*F)
Cbar = ufl.variable(Fbar.T*Fbar)
ICbar = ufl.variable(tr(Cbar))


#U2Fg = element("Lagrange", msh.basix_cell(), degree=deg_u)

U2Fg = quadrature_element(msh.basix_cell(), degree=quadrature_degree, scheme="default")
T0 = basix.ufl.blocked_element(U2Fg, shape=(3, 3))

#F33g = Constant(msh, PETSc.ScalarType( default_scalar_type(0.5)*ufl.pi*x[0]*dt ) )
#F33g = Constant(msh, 0.5*3.1415*x[0]*0.01 )
#F33g = 1.0-0.5*ufl.pi*x[0]*dt
#Fg = ufl.as_tensor( [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,F33g]] )
#Fg = ufl.as_tensor( [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0-0.5*ufl.pi*x[0]*dt]] )


class GrowthFunction():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((3*3, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 1.0
        values[4] = 1.0
        values[8] = 1.0-0.5*3.1415*x[0]*self.t

        return values


# Growth tensor
Fg = Function(functionspace(msh, T0))
Fgg = GrowthFunction(dt)
Fg.interpolate(Fgg)


# Inverse of Fg
FgInv = ufl.variable(inv(Fg))
Jg = ufl.variable(det(Fg))

# Elastic part of deformation gradient
Fe = ufl.variable(F*FgInv)
Je = ufl.variable(det(Fe))
Ce = ufl.variable(Fe.T*Fe)

# First Piola-Kirchhoff stress (only elastic part)
PK1 = Je**(-2/3)*G*(Fe - 1/3*tr(Ce)*inv(Fe.T)) + Je*p*inv(Fe.T)

# Weak form (Jg is to account for the volume change)
Res = inner(PK1, grad(u_test))*Jg*dx + inner((Je - 1 - p*kappa_inv), p_test)*Jg*dx
dRes = derivative(Res, w, dw)

# set up the nonlinear problem
problem = NonlinearProblem(Res, w, bcs, dRes)

# set the solver parameters
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 50
solver.report = True


#  The Krylov solver parameters.
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{option_prefix}ksp_max_it"] = 30
ksp.setFromOptions()


# displacement projection
Vs = functionspace(msh, U2)

u_proj = Function(V2)
u_proj.name = "displacement"

# pressure projection
U1 = element("Lagrange", msh.basix_cell(), 1)

p_proj = Function(functionspace(msh, U1))
p_proj.name = "pressure"


fname = "Rod3D-disp-.pvd"
VTKfile_Disp = io.VTKFile(msh.comm, fname, "w")
VTKfile_Disp.write_mesh(msh)

fname = "Rod3D-pres-.pvd"
VTKfile_Pres = io.VTKFile(msh.comm, fname, "w")
VTKfile_Pres.write_mesh(msh)

# function to write results to XDMF at time t
def writeResults(time_cur, timeStep):

    # Displacement
    u_proj.interpolate(w.sub(0))
    VTKfile_Disp.write_function(u_proj)

    # Displacement
    p_proj.interpolate(w.sub(1))
    VTKfile_Pres.write_function(p_proj)



print("\n----------------------------\n")
print("Simulation has started")
print("\n----------------------------\n")

timeStep = 0
time_cur = 0.0

writeResults(time_cur, timeStep)


# loop over time steps
while ( round(time_cur+dt, 6) <= time_final):
    # Update current time
    time_cur += dt
    timeStep += 1

    print("\n\n Load step = ", timeStep)
    print("     Time      = ", time_cur)

    #F33g.value = PETSc.ScalarType(1.0-0.5*3.1415*x[0]*time_cur)
    #F33g = 1.0-0.5*ufl.pi*x[0]*time_cur
    Fgg.t = time_cur
    Fg.interpolate(Fgg)
    #Fg = ufl.as_tensor( [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0-0.5*3.1415*x[0]*time_cur]] )
    #Fg = ufl.as_tensor( [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0-0.5*ufl.pi*x[0]*dt]] )

    # solution predictor. This will improve convergence.
    w.x.array[:] = 2*w_old.x.array[:] - w_old2.x.array[:]


    # Solve the problem
    num_its = 0
    converged = False
    (num_its, converged) = solver.solve(w)
    
    if converged:
        print(f"Converged in {num_its} iterations.")
    else:
        print(f"Not converged.")

    writeResults(time_cur, timeStep)

    # save variables
    # DOFs
    w_old2.x.array[:] = w_old.x.array[:]
    w_old.x.array[:] = w.x.array[:]
    #w.vector.copy(w_old.vector)Disp


VTKfile_Disp.close()
VTKfile_Pres.close()

