from think_tank import ThinkTank

if __name__ == "__main__":
    DESC = (
        "Use machine learning to develop nanobodies that bind to KP.3 variant "
        "of SARS-CoV-2 spike protein while retaining cross-reactivity."
    )
    lab = ThinkTank(DESC)
    lab.run_team_meeting("Choose initial nanobody design strategy")