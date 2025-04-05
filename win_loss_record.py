import pandas as pd
from datetime import datetime
from fpdf import FPDF

# Function to calculate the worst time periods
def find_worst_times(data, min_consecutive_losses=4):
    max_consecutive_losses = 0
    max_consecutive_wins = 0
    consecutive_losses = 0
    consecutive_wins = 0
    worst_periods = []
    current_losses = []
    start_worst = None

    for i, row in data.iterrows():
        if row['result'] == 'lose':
            consecutive_losses += 1
            consecutive_wins = 0
            current_losses.append(row['time'])  # Track the current consecutive losses

            if consecutive_losses == 1:
                start_worst = row['time']

            if consecutive_losses > max_consecutive_losses:
                max_consecutive_losses = consecutive_losses
        else:  # result == 'win'
            consecutive_wins += 1
            consecutive_losses = 0
            if consecutive_wins > max_consecutive_wins:
                max_consecutive_wins = consecutive_wins

            if len(current_losses) > min_consecutive_losses:
                worst_periods.append((current_losses[0], current_losses[-1]))
            current_losses = []  # Reset the current losses

    # Handle any leftover losses at the end
    if len(current_losses) > min_consecutive_losses:
        worst_periods.append((current_losses[0], current_losses[-1]))

    # Add duration to each period
    worst_periods_with_duration = []
    for start, end in worst_periods:
        start_time = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
        duration = (end_time - start_time).total_seconds() / 60  # in minutes
        worst_periods_with_duration.append((start, end, duration))

    return {
        'max_consecutive_losses': max_consecutive_losses,
        'max_consecutive_wins': max_consecutive_wins,
        'worst_periods': worst_periods_with_duration
    }

# Function to calculate the maximum consecutive losses and wins for today
def find_today_consecutive(data):
    today = datetime.now().strftime('%Y-%m-%d')
    today_data = data[data['time'].str.startswith(today)]  # Filter data for today

    max_consecutive_losses = 0
    max_consecutive_wins = 0
    consecutive_losses = 0
    consecutive_wins = 0

    for i, row in today_data.iterrows():
        if row['result'] == 'lose':
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:  # result == 'win'
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)

    return {
        'max_consecutive_losses_today': max_consecutive_losses,
        'max_consecutive_wins_today': max_consecutive_wins
    }

# Function to generate the PDF
def generate_pdf(color_results, number_results, output_file='results.pdf'):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Gambling Analysis Results', ln=True, align='C')
    pdf.ln(10)

    for section, results in [('Color Analysis', color_results), ('Number Analysis', number_results)]:
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, section, ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.ln(5)

        # Display max consecutive losses and wins
        pdf.cell(0, 10, f"Maximum Consecutive Losses: {results['max_consecutive_losses']}", ln=True)
        pdf.cell(0, 10, f"Maximum Consecutive Wins: {results['max_consecutive_wins']}", ln=True)
        pdf.ln(5)

        # Display max consecutive losses and wins today
        pdf.cell(0, 10, f"Maximum Consecutive Losses Today: {results['max_consecutive_losses_today']}", ln=True)
        pdf.cell(0, 10, f"Maximum Consecutive Wins Today: {results['max_consecutive_wins_today']}", ln=True)
        pdf.ln(5)

        if results['worst_periods']:
            pdf.cell(0, 10, "Worst Time Periods (More than 4 Consecutive Losses):", ln=True)
            for i, (start, end, duration) in enumerate(results['worst_periods'], 1):
                pdf.cell(0, 10, f"{i}. {start} to {end} (Duration: {duration:.2f} mins)", ln=True)
            pdf.ln(5)
        else:
            pdf.cell(0, 10, "Worst Time Periods: None", ln=True)
            pdf.ln(5)

        pdf.ln(5)

    pdf.output(output_file)

# Main logic
def analyze_and_generate_report(color_file, number_file, output_pdf='results.pdf'):
    # Load data
    color_data = pd.read_csv(color_file)
    number_data = pd.read_csv(number_file)

    # Analyze data
    color_results = find_worst_times(color_data)
    number_results = find_worst_times(number_data)

    # Calculate max consecutive losses and wins today
    color_results.update(find_today_consecutive(color_data))
    number_results.update(find_today_consecutive(number_data))

    # Generate PDF
    generate_pdf(color_results, number_results, output_file=output_pdf)

# Example Usage
color_file = 'color_win_loss.csv'
number_file = 'number_win_loss.csv'
output_pdf = 'Analysis_Report.pdf'
analyze_and_generate_report(color_file, number_file, output_pdf)