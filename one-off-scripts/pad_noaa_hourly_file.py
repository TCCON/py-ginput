from argparse import ArgumentParser
from datetime import datetime, timedelta


def main():
    p = ArgumentParser(description='Pad NOAA hourly files with fill values')
    p.add_argument('input_file', help='The NOAA hourly file to extend')
    p.add_argument('output_file', help='Where to write the padded file')
    p.add_argument('--creation-year-month', type=_ymtuple, 
                   help='Year and month (in YYYY-MM format) expected for the creation time in the input file. '
                        'If omitted, then the current year and month are used. This is used to check that the '
                        'creation time attribute in the file header is sensible.')
    p.add_argument('--end-datetime', type=_parsedt, 
                   help='Controls when the appended fill values stop. If not given, the default is to use '
                        'January 1st of the next year, e.g. if run in May 2025, this will be set to midnight, 1 Jan 2026. '
                        'That will cause the last time in the file to be 2300Z, 31 Dec 2025. If given, this must be in '
                        'YYYY-MM-DDTHH:MM:SS format, e.g. "2026-01-01T00:00:00".')

    clargs = vars(p.parse_args())
    driver(**clargs)


def _ymtuple(s):
    ystr, mstr = s.split('-')
    return (int(ystr), int(mstr))


def _parsedt(s):
    return datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')



def driver(input_file, output_file, creation_year_month=None, end_datetime=None):
    with open(input_file) as fin, open(output_file, 'w') as fout:
        check_creation_date(fin, creation_year_month)
        fin.seek(0)
        append_fills(fin, fout, end_datetime=end_datetime)


def check_creation_date(input_handle, target_year_month=None):
    if target_year_month is None:
        now = datetime.now()
        target_year, target_month = (now.year, now.month)
    else:
        target_year, target_month = target_year_month

    for line in input_handle:
        if line.startswith('# description_creation-time:'):
            creation_time = datetime.strptime(line.split(':', maxsplit=1)[1].strip(), '%Y-%m-%d %H:%M:%S')
            if creation_time.year != target_year or creation_time.month != target_month:
                raise RuntimeError(
                    f'Creation time in the input file ({creation_time}) does not have the expected year and month '
                    f'({target_year:04d}-{target_month:02d}). If running this script in a month other than the one '
                    'the file was expected to be created in, use the --creation-year-month argument to set the expected values'
                )
            else:
                print(f'Creation time of {creation_time} is as expected')
            return


def append_fills(input_handle, output_handle, end_datetime=None,
                 year_col='year', month_col='month', day_col='day', hour_col='hour',
                 value_col='value', sd_col='std_dev', value_fill='-999.990', sd_fill='-99.990'):
    data_field_line = None
    for line in input_handle:
        if line.startswith('# data_fields:'):
            data_field_line = line
        output_handle.write(line)
    last_line = line
    if data_field_line is None:
        raise IOError('Failed to find a line containing the data field names')

    data_fields = data_field_line.split(':')[1].split()
    col_inds = _find_field_starts(last_line)
    # I hate using addition operators to build strings,
    # but this is a reasonable use for it.
    fmt_str = '' 
    defaults = dict()
    for icol, col in enumerate(data_fields):
        j1 = col_inds[icol]
        j2 = col_inds[icol+1]
        width = j2 - j1
        if col == year_col:
            leading_space = ' ' * (width - 4)
            fmt_str += leading_space + f'{{{col}:04d}}'
        elif col in {month_col, day_col, hour_col}:
            leading_space = ' ' * (width - 2)
            fmt_str += leading_space + f'{{{col}:02d}}'
        else:
            fmt_str += f'{{{col}:>{width}}}'
        defaults[col] = last_line[j1:j2]
    fmt_str += '\n'

    # Since defaults has the values for the final line of the file,
    # we can use it to figure out the time range we need
    year = int(defaults[year_col])
    month = int(defaults[month_col])
    day = int(defaults[day_col])
    hour = int(defaults[hour_col])
    curr_datetime = datetime(year, month, day, hour, 0, 0)
    curr_datetime += timedelta(hours=1)
    if end_datetime is None:
        end_datetime = datetime(year+1, 1, 1)

    nadded = 0
    while curr_datetime < end_datetime:
        new_values = defaults.copy()
        new_values.update({
            year_col: curr_datetime.year,
            month_col: curr_datetime.month,
            day_col: curr_datetime.day,
            hour_col: curr_datetime.hour,
            value_col: value_fill,
            sd_col: sd_fill
        })
        new_line = fmt_str.format(**new_values)
        output_handle.write(new_line)
        curr_datetime += timedelta(hours=1)
        nadded += 1

    print(f'Added {nadded} lines with fill values for CO2 and its standard deviation')




def _find_field_starts(line):
    starts = [0]
    next_whitespace = not line[0].isspace()
    for i, c in enumerate(line):
        if next_whitespace and c.isspace():
            starts.append(i)
            next_whitespace = False
        elif not next_whitespace and not c.isspace():
            next_whitespace = True

    # 2 cases: (1) line ends on a non-whitespace character, meaning we need
    # to add the line length as the final "start" to know the width of the last field.
    # (2) line has whitespace, possibly just a newline, after the last field, in which
    # case we do not need to add a final "start"
    if next_whitespace:
        # case 1
        starts.append(len(line))
    return starts


if __name__ == '__main__':
    main()
