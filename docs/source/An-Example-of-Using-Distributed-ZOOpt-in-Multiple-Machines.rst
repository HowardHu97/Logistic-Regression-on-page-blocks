In this example, we have four machines: one for the client (M1), one for
control server (M2) and two for evaluation servers (M3 and M4).

Launch Servers
~~~~~~~~~~~~~~

Launch a control server
^^^^^^^^^^^^^^^^^^^^^^^

Write a simple start\_control\_server.py and upload it to M2. `(Example
code in
ZOOsrv) <https://github.com/eyounx/ZOOsrv/blob/master/example/start_control_server.py>`__

    start\_control\_server.py:

.. code:: python

    from zoosrv import control_server
    control_server.start(20000)

where the parameter 20000 is the listening port of the control server.
Then, run the codes as in command line

::

    $ python start_control_server.py

Launch evaluation servers
^^^^^^^^^^^^^^^^^^^^^^^^^

Write a start\_evaluation\_server.py, including the following codes
`(Example code in
ZOOsrv) <https://github.com/eyounx/ZOOsrv/blob/master/example/start_evaluation_server.py>`__

.. code:: python

    from zoosrv import evaluation_server
    evaluation_server.start("evaluation_server.cfg")

where evaluation\_server.cfg is the configuration file.

Then, write the evaluation\_server.cfg file including the following
lines: `(Example code in
ZOOsrv) <https://github.com/eyounx/ZOOsrv/blob/master/example/evaluation_server.cfg>`__

::

    [evaluation server]
    shared fold = /path/to/project/ZOOsrv/example/objective_function/
    control server ip_port = 192.168.0.103:20000
    evaluation processes = 10
    starting port = 60003
    ending port = 60020

where ``shared fold`` is the fold storing the objective function files.
``control server ip_port`` is the address of the control server, and the
last three lines state we want to start 10 evaluation processes by
choosing 10 available ports from 60003 to 60020. Notice that users can
write different configuration files for different machines.

Then, upload start\_evaluation\_server.py, evaluation\_server.cfg and
the directory including objective function file (defined in the next
part) to M3 and M4.

Finally, launch evaluation servers respectively in M3 and M4.

::

    $ python start_evaluation_server.py

Perform Optimization
~~~~~~~~~~~~~~~~~~~~

We try to optimize the Ackley function.

Define the objective function in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Write ``fx.py`` including the following codes `(Example code in
ZOOsrv) <https://github.com/eyounx/ZOOsrv/blob/master/example/objective_function/fx.py>`__

.. code:: python

    import numpy as np
    def ackley(solution):
        x = solution.get_x()
        bias = 0.2
        value = -20 * np.exp(-0.2 * np.sqrt(sum([(i - bias) * (i - bias) for i in x]) / len(x))) - \
                np.exp(sum([np.cos(2.0*np.pi*(i-bias)) for i in x]) / len(x)) + 20.0 + np.e
        return value

where ``shared fold`` is the directory the ``fx.py`` stores.

Write client code in Julia
^^^^^^^^^^^^^^^^^^^^^^^^^^

Write ``client.jl`` including the following codes `(Example code in
ZOOsrv) <https://github.com/eyounx/ZOOjl.jl/blob/master/example/client.jl>`__

.. code:: julia

    using ZOOclient
    using PyPlot

    # define a Dimension object
    dim_size = 100
    dim_regs = [[-1, 1] for i = 1:dim_size]
    dim_tys = [true for i = 1:dim_size]
    mydim = Dimension(dim_size, dim_regs, dim_tys)
    # define an Objective object
    obj = Objective(mydim)

    # define a Parameter Object, the five parameters are indispensable.
    # budget:  number of calls to the objective function
    # evalueation_server_num: number of evaluation cores user requires
    # control_server_ip_port: the ip:port of the control server
    # objective_file: objective funtion is defined in this file
    # func: name of the objective function
    par = Parameter(budget=10000, evaluation_server_num=20, control_server_ip_port="192.168.0.103:20000",
        objective_file="fx.py", func="ackley")

    # perform optimization
    sol = zoo_min(obj, par)
    # print the Solution object
    sol_print(sol)

    # visualize the optimization progress
    history = get_history_bestsofar(obj)
    plt[:plot](history)
    plt[:savefig]("figure.png")

Upload this file to the client machine (M1) and run it to perform the
optimization

::

    $ ./julia -p 4 /absolute/path/to/your/file/client.jl

where ``julia -p n`` provides ``n`` processes for the client on the
local machine. Generally it makes sense for ``n`` to equal the number of
CPU cores on the machine.

For a few seconds, the optimization is done and we will get the result.

 .. image::https://github.com/eyounx/ZOOjl.jl/blob/master/img/result.png?raw=true

Visualized optimization progress looks like:

.. image::https://github.com/eyounx/ZOOjl.jl/blob/master/img/figure.png?raw=true
       
