import { useState, useCallback, useEffect } from 'react';

import axios from 'axios';
import { Box, Typography } from '@mui/material';

import { DashboardContent } from 'src/layouts/dashboard';
import { _users } from 'src/_mock';
import { emptyRows, applyFilter, getComparator } from '../utils';

import type { UserProps } from '../user-table-row';
// ----------------------------------------------------------------------

export function UserView() {
  const table = useTable();

  const [filterName, setFilterName] = useState('');

  const dataFiltered: UserProps[] = applyFilter({
    inputData: _users,
    comparator: getComparator(table.order, table.orderBy),
    filterName,
  });

  const notFound = !dataFiltered.length && !!filterName;

  const [file, setFile] = useState<FileList | null>(null);
  const [progress, setProgress] = useState({started: false, pc: 0});
  const [msg, setMsg] = useState('');

  function handleUpload() { 
    const fd = new FormData()
    if (file && file.length > 0) { // Ensure file is not null and has at least one file
        fd.append('file', file[0]); // Append the first file in the FileList
    } else {
        setMsg('Please select a file to upload.'); // Handle the case when no file is selected
        return; // Prevent further execution
    }

    fetch('http://127.0.0.1:5000/test1', {
        method: "POST",
        body: fd
    })
    .then(res => res.json())
    .then(data => console.log(data))

  }

  return (
    <DashboardContent>
      <Box display="flex" alignItems="center" mb={5}>
        <Typography variant="h4" flexGrow={1}>
          Financial Tips 💲💸
        </Typography>
      </Box>
      <div>
        <h1>Select User Details</h1>
        <input type="file" onChange={(e) => setFile(e.target.files)} />
        <button type="button" onClick={ handleUpload}>Upload File</button>
        { progress.started && <progress max="100" value={progress.pc} />}
        { msg && <span>{msg}</span> }
      </div>
    </DashboardContent>
  );
}

// ----------------------------------------------------------------------

export function useTable() {
  const [page, setPage] = useState(0);
  const [orderBy, setOrderBy] = useState('name');
  const [rowsPerPage, setRowsPerPage] = useState(5);
  const [selected, setSelected] = useState<string[]>([]);
  const [order, setOrder] = useState<'asc' | 'desc'>('asc');

  const onSort = useCallback(
    (id: string) => {
      const isAsc = orderBy === id && order === 'asc';
      setOrder(isAsc ? 'desc' : 'asc');
      setOrderBy(id);
    },
    [order, orderBy]
  );

  const onSelectAllRows = useCallback((checked: boolean, newSelecteds: string[]) => {
    if (checked) {
      setSelected(newSelecteds);
      return;
    }
    setSelected([]);
  }, []);

  const onSelectRow = useCallback(
    (inputValue: string) => {
      const newSelected = selected.includes(inputValue)
        ? selected.filter((value) => value !== inputValue)
        : [...selected, inputValue];

      setSelected(newSelected);
    },
    [selected]
  );

  const onResetPage = useCallback(() => {
    setPage(0);
  }, []);

  const onChangePage = useCallback((event: unknown, newPage: number) => {
    setPage(newPage);
  }, []);

  const onChangeRowsPerPage = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      setRowsPerPage(parseInt(event.target.value, 10));
      onResetPage();
    },
    [onResetPage]
  );

  return {
    page,
    order,
    onSort,
    orderBy,
    selected,
    rowsPerPage,
    onSelectRow,
    onResetPage,
    onChangePage,
    onSelectAllRows,
    onChangeRowsPerPage,
  };
}
