PLOTLY_FONT = "#111827"
PLOTLY_BG = "#FFFFFF"

def style_plotly(fig):

    fig.update_layout(

        paper_bgcolor=PLOTLY_BG,
        plot_bgcolor=PLOTLY_BG,

        font=dict(
            family="Inter",
            size=15,
            color=PLOTLY_FONT
        ),

        title_font=dict(
            size=22,
            color=PLOTLY_FONT
        ),

        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_color="#111827"
        ),

        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor="#CBD5E1",
            tickfont=dict(
                size=13,
                color="#111827"
            ),
            title_font=dict(
                size=15,
                color="#111827"
            )
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="#E5E7EB",
            tickfont=dict(
                size=13,
                color="#111827"
            ),
            title_font=dict(
                size=15,
                color="#111827"
            )
        ),

        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(
                size=14,
                color="#111827"
            ),
            title=dict(
                font=dict(
                    color="#111827"
                )
            )
        ),

        margin=dict(
            l=60,
            r=30,
            t=70,
            b=60
        )
    )

    return fig