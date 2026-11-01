from prefect import flow, task


@task
def check_setup():
    return "Prefect is working!"


@flow(name="homework-3-check")
def check_flow():
    message = check_setup()
    return message


if __name__ == "__main__":
    print(check_flow())