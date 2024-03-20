import chainlit as cl
import time

INITIAL_MSG = """Demo for TaskList issue #667"""

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content=INITIAL_MSG,
                     author="Announcer").send()


@cl.on_message
async def run_conversation(message: cl.Message):
    # Create the TaskList
    task_list = cl.TaskList()
    task_list.status = "Running..."

    # Agent 1 task
    initial_task = cl.Task("Processing query...", cl.TaskStatus.RUNNING)
    await task_list.add_task(initial_task)
    await task_list.send()
    # Agent 1 msg
    agent_1_msg = cl.Message(content="", author="Agent 1")
    await agent_1_msg.send()
    await cl.sleep(0)   # to show the loader
    
    # simulate async operation
    await cl.sleep(2)
    # update the first task
    initial_task.status = cl.TaskStatus.DONE
    await task_list.send()

    # update Agent 1 msg
    agent_1_msg.content = "Agent 1 is done"
    await agent_1_msg.update()
    
    # now add more tasks
    new_tasks: list[cl.Task] = []
    for task_label in ['Processing results...', 'Extracting information...']:
        t = cl.Task(task_label, cl.TaskStatus.READY)
        new_tasks.append(t)
        await task_list.add_task(t)
    
    # simulate running each task
    for i, task in enumerate(new_tasks):
        # update task
        task.status = cl.TaskStatus.RUNNING
        await task_list.send()
        
        # set a loading message
        agent_msg = cl.Message(content="", author=f'Agent {i+2}')
        await agent_msg.send()
        await cl.sleep(0)

        if i == 0:
            # simulate streaming
            dummy_text = 'Lorem ipsum dolor sit amet ' * 5
            await agent_msg.remove()
            msg = cl.Message(author='Agent 2', content="")
            for token in dummy_text.split(' '):
                await msg.stream_token(token + ' ')
                await cl.sleep(0.3)
            await msg.send()

        elif i == 1:
            # simulate async
            await cl.sleep(2)
            # simulate sync
            time.sleep(2)

            # update msg
            agent_msg.content = f'Ran Agent 3'
            await agent_msg.update()
                
        # update task
        task.status = cl.TaskStatus.DONE
        await task_list.send()
    
    # update task list finally
    task_list.status = "Done"
    await task_list.send()