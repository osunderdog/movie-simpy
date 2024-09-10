"""
Movies line

- wants to wait at most 10 minutes

"""
import logging
import random
import statistics

from pydantic import BaseModel, Field
from datetime import timedelta
from itertools import groupby, product

import simpy

logging.basicConfig(level=logging.DEBUG)


class Theater:
    def __init__(self, env, num_cashiers, num_servers, num_ushers):
        self.env = env
        self.cashier = simpy.Resource(env, num_cashiers)
        self.server = simpy.Resource(env, num_servers)
        self.usher = simpy.Resource(env, num_ushers)

    def purchase_ticket(self, moviegoer):
        yield self.env.timeout(random.randint(1, 3))

    def check_ticket(self, moviegoer):
        yield self.env.timeout(3 / 60)

    def sell_food(self, moviegoer):
        yield self.env.timeout(random.randint(1, 5))


def go_to_movies(env, moviegoer, theater, wait_times):
    # Moviegoer arrives at the theater
    arrival_time = env.now

    with theater.cashier.request() as request:
        yield request
        yield env.process(theater.purchase_ticket(moviegoer))

    with theater.usher.request() as request:
        yield request
        yield env.process(theater.check_ticket(moviegoer))

    if random.choice([True, False]):
        with theater.server.request() as request:
            yield request
            yield env.process(theater.sell_food(moviegoer))

    # Moviegoer heads into the theater
    wait_times.append(env.now - arrival_time)


def run_theater(env, num_cashiers, num_servers, num_ushers, wait_times):
    theater = Theater(env, num_cashiers, num_servers, num_ushers)

    for moviegoer in range(3):
        env.process(go_to_movies(env, moviegoer, theater, wait_times))

    #why 27?
    for _ in range(27):
        yield env.timeout(0.20)  # Wait a bit before generating new moviegoer

        # Almost done!...
        moviegoer += 1
        env.process(go_to_movies(env, moviegoer, theater, wait_times))

    return wait_times



def run_simulation(num_cashiers, num_servers, num_ushers):
    wait_times = []
    # Run the simulation
    env = simpy.Environment()
    env.process(run_theater(env, num_cashiers, num_servers, num_ushers, wait_times))
    env.run()
    return wait_times



class EmployeeConfig(BaseModel):
    num_cashiers: int = Field(description="Number of cachiers available")
    num_servers: int = Field(description="Number of servers that are available.")
    num_ushers: int = Field(description="Number ofushers available")
    average_time: float = Field(default=None, description="how long it took to run people through the system")

    #TODO keep times and the use property to calculate average.

    @property
    def total_employees(self):
        return sum((self.num_cashiers, self.num_servers, self.num_ushers))

    def calculate_average_wait_time_for_config(self, num_simulations):
        self.average_time = timedelta(minutes=statistics.mean(run_simulation(self.num_cashiers, self.num_servers, self.num_ushers))).total_seconds()

    @classmethod
    def generate_employee_config(cls, max_employees):
        """Generate the parameters for an EmployeeConfig. number of places is a little strange from the docs: For example, product(A, repeat=4) means the same as product(A, A, A, A).  So in this case, build a tuple of three items."""
        num_places = 3
        return [cls(num_cashiers=config[0], num_servers=config[1], num_ushers=config[2]) for config in product(range(1, max_employees + 1), repeat=num_places) if sum(config) <= max_employees]


def main():
    # Setup
    random.seed(42)

    # Run simulation

    max_employees = 10
    num_simulations = 10
    emp_config_list = EmployeeConfig.generate_employee_config(max_employees)

    # Run the simulation and retrieve the average wait time from each run back into the EmployeeConfig.
    [ec.calculate_average_wait_time_for_config(num_simulations) for ec in emp_config_list]

    def total_employees_key(e):
        return e.total_employees

    emp_config_list.sort(key=lambda e: e.total_employees)
    for k, g in groupby(emp_config_list, total_employees_key):
        if True:
            g = list(g)
            for group_item in g:
                logging.debug(group_item)
        best_config = min(g, key=lambda e: e.average_time)
        logging.info(f"For {k} employees, best config is {best_config}\n")



if __name__ == "__main__":
    main()
