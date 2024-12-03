import gradio as gr


def launch_interface(
    logo_gen_app: (...)
):

    logo_gen_interface = gr.Interface(
        fn=logo_gen_app,
        inputs=[
            gr.Checkbox(label="Minimal B&W", value=True),
            gr.Image(label="Image", type="pil")
        ],
        outputs=[
            gr.Image(label="Logo", type="pil")
        ],
        allow_flagging="never"
    )

    interface = gr.TabbedInterface(
        interface_list=[
            logo_gen_interface
        ],
        tab_names=[
            "LogoGen"
        ]
    )

    interface.launch(share=True)