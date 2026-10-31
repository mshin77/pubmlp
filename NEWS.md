# pubmlp 0.6.3

- Cites the page and section each extracted answer comes from.
- Checks per paper whether its text may go to a language model, and where.
- Keeps passwords and keys out of logs.
- Starts faster by reading package versions without loading the packages.

# pubmlp 0.6.2

- Finds a search term when the star sits inside the phrase, not only at the end.
- Colors the matched words inside a screened spreadsheet.

# pubmlp 0.6.1

- Stops screening once a stronger model finds nothing more relevant.
- Adds a setting to choose the stronger model.
- Renames two functions to say what they do.

# pubmlp 0.6.0

- Reads search exports in CSV, Excel, RIS, BibTeX, and Web of Science formats.
- Finds evidence in full text and reports the page it came from.
- Lets a language model screen as a second rater, through a supplied function.
- Saves what each run used, so a result can be traced.
- Fixes counts in the audit trail, the stopping rule, and the record reader.
