# from: https://github.com/tomasmenito/movie-simpy.git
# backtracking from this article:
# https://realpython.com/simpy-simulating-with-python/

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
    """Simulate a theater with some limited resources.
    Cashier: Number of people that can provide a ticket
    Server: Number of people that can vend food.
    Usher: Number of people that can check a ticket and admit them to the movie."""
    def __init__(self, env, num_cashiers, num_servers, num_ushers):
        # Pysim execution environment
        self.env = env
        # the limited number of cashiers that are available in the Theater.
        self.cashier = simpy.Resource(env, num_cashiers)

        # the limited number of servers that are available in the Theater.
        self.server = simpy.Resource(env, num_servers)

        # the limited number of ushers that are available in the Theater.
        self.usher = simpy.Resource(env, num_ushers)

    def purchase_ticket(self, moviegoer):
        """Purchasing a ticket from the theater takes some time.  Yield the random duration needed to purchase a ticket."""
        yield self.env.timeout(random.randint(1, 3))

    def check_ticket(self, moviegoer):
        """Checking the ticket takes some time.  The time taken is fixed at 20 time units"""
        yield self.env.timeout(3 / 60)

    def sell_food(self, moviegoer):
        """Selling food takes some time  Yield the random duration needed to sell food."""
        yield self.env.timeout(random.randint(1, 5))


    @classmethod
    def go_to_movies(cls, env, moviegoer, theater, wait_times):
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

    @classmethod
    def run_theater(cls, env, num_cashiers, num_servers, num_ushers, wait_times):
        theater = Theater(env, num_cashiers, num_servers, num_ushers)

        # Generate 30 movie goers.
        # First three are simultaneously then at a slow trickle?
        for moviegoer in range(3):
            env.process(cls.go_to_movies(env, moviegoer, theater, wait_times))

        #why 27?
        for _ in range(27):
            yield env.timeout(0.20)  # Wait a bit before generating new moviegoer

            # Almost done!...
            moviegoer += 1
            env.process(cls.go_to_movies(env, moviegoer, theater, wait_times))

        return wait_times


    @classmethod
    def run_simulation(cls, num_cashiers, num_servers, num_ushers):
        wait_times = []
        # Run the simulation
        env = simpy.Environment()
        env.process(cls.run_theater(env, num_cashiers, num_servers, num_ushers, wait_times))
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
        self.average_time = timedelta(minutes=statistics.mean(Theater.run_simulation(self.num_cashiers, self.num_servers, self.num_ushers))).total_seconds()

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

    emp_config_list.sort(key=lambda e: e.total_employees)
    for k, g in groupby(emp_config_list, lambda e: e.total_employees):
        if True:
            g = list(g)
            for group_item in g:
                logging.debug(group_item)
        best_config = min(g, key=lambda e: e.average_time)
        logging.info(f"For {k} employees, best config is {best_config}\n")



if __name__ == "__main__":
    main()
